"""
LaBraM (Large Brain Model) Wrapper for EEG Classification.

This module provides a wrapper around the pretrained LaBraM model for fine-tuning
on downstream EEG classification tasks like motor imagery.

LaBraM repository: https://github.com/935963004/LaBraM

Usage:
    from src.labram_wrapper import LaBraMClassifier, load_labram_backbone
    
    backbone = load_labram_backbone(checkpoint_path="checkpoints/labram-base.pth")
    model = LaBraMClassifier(
        backbone=backbone,
        n_classes=2,
        n_channels=22,
        n_samples=750,
        unfreeze_last_n_layers=8
    )
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def _try_import_labram():
    """
    Attempt to import LaBraM from various locations.
    
    LaBraM can be installed via:
        1. pip install git+https://github.com/935963004/LaBraM.git
        2. Cloning the repo and adding to sys.path
        
    Returns:
        The LaBraM module or raises ImportError with instructions.
    """
    # Try direct import (if installed via pip)
    try:
        import labram
        return labram
    except ImportError:
        pass
    
    # Try importing from a cloned repo in common locations
    possible_paths = [
        Path("LaBraM"),  # Current directory
        Path("../LaBraM"),  # Parent directory
        Path.home() / "LaBraM",  # Home directory
        Path("/content/LaBraM"),  # Colab default
    ]
    
    for path in possible_paths:
        if path.exists():
            sys.path.insert(0, str(path.resolve()))
            try:
                import labram
                return labram
            except ImportError:
                sys.path.pop(0)
    
    raise ImportError(
        "LaBraM not found. Install via:\n"
        "  pip install git+https://github.com/935963004/LaBraM.git\n"
        "Or clone the repository:\n"
        "  git clone https://github.com/935963004/LaBraM.git"
    )


def load_labram_backbone(
    checkpoint_path: str,
    model_name: str = "labram_base_patch200_200",
) -> nn.Module:
    """
    Load a pretrained LaBraM backbone.
    
    Args:
        checkpoint_path: Path to the pretrained checkpoint (.pth file)
        model_name: LaBraM model variant name (e.g., "labram_base_patch200_200")
        
    Returns:
        Pretrained LaBraM backbone model
        
    Note:
        This function may need adjustment based on LaBraM's actual API.
        The model_name corresponds to LaBraM's model registry.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Download pretrained weights from the LaBraM repository."
        )
    
    labram = _try_import_labram()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Try to instantiate LaBraM model
    # Note: Adjust based on actual LaBraM API
    try:
        # Method 1: Using model registry (common pattern)
        if hasattr(labram, "create_model"):
            backbone = labram.create_model(model_name, pretrained=False)
        elif hasattr(labram, model_name):
            backbone = getattr(labram, model_name)()
        elif hasattr(labram, "LaBraM"):
            backbone = labram.LaBraM()
        elif hasattr(labram.models, "labram_base_patch200_200"):
            backbone = labram.models.labram_base_patch200_200()
        else:
            # Fallback: try to find any model class
            for attr_name in dir(labram):
                attr = getattr(labram, attr_name)
                if isinstance(attr, type) and issubclass(attr, nn.Module):
                    backbone = attr()
                    break
            else:
                raise AttributeError("Could not find LaBraM model class")
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate LaBraM model '{model_name}': {e}\n"
            "Check the LaBraM repository for correct model names and API."
        )
    
    # Load pretrained weights
    # Strip 'module.' prefix if present (from DataParallel)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Try to load weights (allow missing keys for classification head)
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Note: Missing keys in checkpoint (likely classification head): {len(missing)} keys")
    if unexpected:
        print(f"Note: Unexpected keys in checkpoint: {len(unexpected)} keys")
    
    return backbone


class LaBraMClassifier(nn.Module):
    """
    LaBraM-based classifier for EEG motor imagery.
    
    Wraps a pretrained LaBraM backbone with:
    - Input adaptation layer (matches your pipeline's [B, 1, C, T] format)
    - Classification head for n_classes
    - Layer freezing for efficient fine-tuning
    
    Args:
        backbone: Pretrained LaBraM model
        n_classes: Number of output classes
        n_channels: Number of EEG channels in your data
        n_samples: Number of time samples in your data
        unfreeze_last_n_layers: Number of transformer layers to unfreeze (default: 8)
        hidden_dim: Hidden dimension of classification head (default: 256)
        dropout: Dropout rate for classification head (default: 0.5)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int,
        n_channels: int,
        n_samples: int,
        unfreeze_last_n_layers: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        
        # Determine backbone's expected input and output dimensions
        # This may need adjustment based on actual LaBraM architecture
        self._infer_backbone_dims()
        
        # Input adapter: converts [B, 1, C, T] to backbone's expected format
        # LaBraM likely expects [B, C, T] or [B, n_patches, patch_dim]
        self.input_adapter = self._build_input_adapter()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.backbone_out_dim),
            nn.Dropout(dropout),
            nn.Linear(self.backbone_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        
        # Apply layer freezing
        self.freeze_except_last_n_layers(unfreeze_last_n_layers)
    
    def _infer_backbone_dims(self):
        """Infer backbone input/output dimensions."""
        # Try to get embed_dim from backbone
        if hasattr(self.backbone, "embed_dim"):
            self.backbone_out_dim = self.backbone.embed_dim
        elif hasattr(self.backbone, "num_features"):
            self.backbone_out_dim = self.backbone.num_features
        elif hasattr(self.backbone, "config") and hasattr(self.backbone.config, "hidden_size"):
            self.backbone_out_dim = self.backbone.config.hidden_size
        else:
            # Default for LaBraM-Base (typically 768 or 200)
            self.backbone_out_dim = 200
            print(f"Warning: Could not infer backbone dim, using default: {self.backbone_out_dim}")
        
        # Try to get expected input channels
        if hasattr(self.backbone, "in_chans"):
            self.backbone_in_chans = self.backbone.in_chans
        else:
            self.backbone_in_chans = None
    
    def _build_input_adapter(self) -> nn.Module:
        """
        Build adapter to convert input from [B, 1, C, T] to backbone's format.
        
        LaBraM likely expects one of:
        - [B, C, T]: Standard EEG format
        - [B, n_patches, patch_dim]: Tokenized format
        """
        # For now, just squeeze the extra dimension
        # May need to add channel/time interpolation if dimensions don't match
        return nn.Identity()
    
    def _get_transformer_blocks(self):
        """
        Get list of transformer blocks from the backbone.
        
        Returns:
            List of (name, module) tuples for transformer blocks
        """
        blocks = []
        
        # Common attribute names for transformer blocks
        block_attr_names = ["blocks", "layers", "encoder", "transformer"]
        
        for attr_name in block_attr_names:
            if hasattr(self.backbone, attr_name):
                block_container = getattr(self.backbone, attr_name)
                if isinstance(block_container, nn.ModuleList):
                    blocks = [(f"{attr_name}.{i}", block) for i, block in enumerate(block_container)]
                    break
                elif isinstance(block_container, nn.Sequential):
                    blocks = [(f"{attr_name}.{i}", block) for i, block in enumerate(block_container)]
                    break
        
        return blocks
    
    def freeze_except_last_n_layers(self, n: int):
        """
        Freeze all backbone parameters except the last n transformer layers.
        
        The classification head is always trainable.
        
        Args:
            n: Number of transformer layers to keep unfrozen (from the end)
        """
        # First, freeze everything in backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get transformer blocks
        blocks = self._get_transformer_blocks()
        
        if blocks:
            # Unfreeze last n blocks
            n_blocks = len(blocks)
            unfreeze_from = max(0, n_blocks - n)
            
            for idx, (name, block) in enumerate(blocks):
                if idx >= unfreeze_from:
                    for param in block.parameters():
                        param.requires_grad = True
            
            print(f"Froze {unfreeze_from}/{n_blocks} transformer blocks, unfroze last {min(n, n_blocks)}")
        else:
            # Fallback: unfreeze last n% of parameters by name
            all_params = list(self.backbone.named_parameters())
            n_params = len(all_params)
            unfreeze_from = max(0, n_params - (n_params * n // 12))  # Rough estimate
            
            for idx, (name, param) in enumerate(all_params):
                if idx >= unfreeze_from:
                    param.requires_grad = True
            
            print(f"Warning: Could not find transformer blocks. Unfroze last {n_params - unfreeze_from} params")
        
        # Ensure classifier is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        # Also unfreeze layer norm / final norm if present
        for name, module in self.backbone.named_modules():
            if "norm" in name.lower() and any(k in name.lower() for k in ["final", "last", "fc_norm"]):
                for param in module.parameters():
                    param.requires_grad = True
    
    def get_trainable_params(self):
        """Get number of trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, 1, C, T]
            
        Returns:
            Logits of shape [B, n_classes]
        """
        # Remove extra dimension: [B, 1, C, T] -> [B, C, T]
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        
        # Apply input adapter
        x = self.input_adapter(x)
        
        # Forward through backbone
        # LaBraM likely returns features from CLS token or pooled features
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
        
        # Handle different output formats
        if isinstance(features, tuple):
            features = features[0]
        
        # If features are sequential [B, T, D], take CLS token or mean pool
        if features.dim() == 3:
            # Option 1: CLS token (first position)
            # features = features[:, 0]
            # Option 2: Mean pooling (more robust)
            features = features.mean(dim=1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


def build_labram_classifier(
    checkpoint_path: str,
    n_classes: int,
    n_channels: int,
    n_samples: int,
    model_name: str = "labram_base_patch200_200",
    unfreeze_last_n_layers: int = 8,
    dropout: float = 0.5,
) -> LaBraMClassifier:
    """
    Convenience function to build a LaBraM classifier.
    
    Args:
        checkpoint_path: Path to pretrained weights
        n_classes: Number of output classes
        n_channels: Number of EEG channels
        n_samples: Number of time samples
        model_name: LaBraM model variant
        unfreeze_last_n_layers: Layers to unfreeze for fine-tuning
        dropout: Dropout rate
        
    Returns:
        Configured LaBraMClassifier ready for fine-tuning
    """
    backbone = load_labram_backbone(checkpoint_path, model_name)
    
    classifier = LaBraMClassifier(
        backbone=backbone,
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        unfreeze_last_n_layers=unfreeze_last_n_layers,
        dropout=dropout,
    )
    
    trainable, total = classifier.get_trainable_params()
    print(f"LaBraMClassifier: {trainable:,} / {total:,} params trainable ({100*trainable/total:.1f}%)")
    
    return classifier

