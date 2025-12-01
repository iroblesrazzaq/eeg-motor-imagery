"""
EEG Data Augmentation Transforms.

These augmentations are critical for cross-subject generalization with limited training data.
They simulate natural variability in EEG recordings and prevent overfitting to subject-specific patterns.

Usage:
    transform = Compose([
        GaussianNoise(std=0.1),
        TimeShift(max_shift=25),
        ChannelDropout(p=0.1),
    ])
    augmented = transform(eeg_tensor)
"""

import random
from typing import List, Optional

import torch


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x
    
    def __repr__(self):
        return f"Compose({self.transforms})"


class GaussianNoise:
    """Add Gaussian noise to the signal.
    
    Simulates sensor noise and makes the model robust to signal variations.
    This is the most important augmentation for EEG generalization.
    
    Args:
        std: Standard deviation of noise (relative to signal std)
        p: Probability of applying the transform
    """
    
    def __init__(self, std: float = 0.1, p: float = 0.5):
        self.std = std
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        
        # Scale noise relative to signal magnitude
        noise_std = self.std * x.std()
        noise = torch.randn_like(x) * noise_std
        return x + noise
    
    def __repr__(self):
        return f"GaussianNoise(std={self.std}, p={self.p})"


class TimeShift:
    """Randomly shift the signal in time (circular shift).
    
    Simulates timing jitter in motor imagery onset.
    
    Args:
        max_shift: Maximum shift in samples (positive or negative)
        p: Probability of applying the transform
    """
    
    def __init__(self, max_shift: int = 25, p: float = 0.5):
        self.max_shift = max_shift
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        
        # Circular shift along time dimension (last dim)
        return torch.roll(x, shifts=shift, dims=-1)
    
    def __repr__(self):
        return f"TimeShift(max_shift={self.max_shift}, p={self.p})"


class ChannelDropout:
    """Randomly zero out entire channels.
    
    Makes the model robust to noisy/missing channels and prevents
    over-reliance on specific electrode locations.
    
    Args:
        p_drop: Probability of dropping each channel
        p: Probability of applying the transform at all
    """
    
    def __init__(self, p_drop: float = 0.1, p: float = 0.5):
        self.p_drop = p_drop
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        
        # x shape: [1, C, T] or [C, T]
        if x.dim() == 3:
            n_channels = x.shape[1]
            mask = torch.rand(1, n_channels, 1) > self.p_drop
        else:
            n_channels = x.shape[0]
            mask = torch.rand(n_channels, 1) > self.p_drop
        
        return x * mask.float().to(x.device)
    
    def __repr__(self):
        return f"ChannelDropout(p_drop={self.p_drop}, p={self.p})"


class TimeMask:
    """Randomly mask (zero out) contiguous time segments.
    
    Similar to SpecAugment for speech - prevents overfitting to specific
    temporal patterns and improves robustness.
    
    Args:
        max_mask_length: Maximum length of time mask in samples
        n_masks: Number of masks to apply
        p: Probability of applying the transform
    """
    
    def __init__(self, max_mask_length: int = 50, n_masks: int = 2, p: float = 0.5):
        self.max_mask_length = max_mask_length
        self.n_masks = n_masks
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        
        x = x.clone()
        T = x.shape[-1]
        
        for _ in range(self.n_masks):
            mask_length = random.randint(1, min(self.max_mask_length, T // 4))
            start = random.randint(0, T - mask_length)
            x[..., start:start + mask_length] = 0
        
        return x
    
    def __repr__(self):
        return f"TimeMask(max_mask_length={self.max_mask_length}, n_masks={self.n_masks}, p={self.p})"


class AmplitudeScale:
    """Randomly scale signal amplitude.
    
    Simulates variations in signal strength across subjects/sessions.
    
    Args:
        scale_range: Tuple of (min_scale, max_scale)
        p: Probability of applying the transform
    """
    
    def __init__(self, scale_range: tuple = (0.8, 1.2), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        
        scale = random.uniform(*self.scale_range)
        return x * scale
    
    def __repr__(self):
        return f"AmplitudeScale(scale_range={self.scale_range}, p={self.p})"


class ChannelPermute:
    """Randomly permute a subset of channels.
    
    Prevents the model from learning rigid spatial patterns and
    encourages learning more robust spatial features.
    
    Args:
        n_permute: Number of channel pairs to swap
        p: Probability of applying the transform
    """
    
    def __init__(self, n_permute: int = 2, p: float = 0.3):
        self.n_permute = n_permute
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        
        x = x.clone()
        
        # Get channel dimension
        if x.dim() == 3:
            n_channels = x.shape[1]
            for _ in range(self.n_permute):
                if n_channels < 2:
                    break
                i, j = random.sample(range(n_channels), 2)
                x[:, [i, j], :] = x[:, [j, i], :]
        else:
            n_channels = x.shape[0]
            for _ in range(self.n_permute):
                if n_channels < 2:
                    break
                i, j = random.sample(range(n_channels), 2)
                x[[i, j], :] = x[[j, i], :]
        
        return x
    
    def __repr__(self):
        return f"ChannelPermute(n_permute={self.n_permute}, p={self.p})"


def build_augmentation_from_config(aug_config: Optional[dict]) -> Optional[Compose]:
    """Build augmentation pipeline from config dictionary.
    
    Args:
        aug_config: Dictionary with augmentation settings, e.g.:
            {
                "gaussian_noise": {"std": 0.1, "p": 0.5},
                "time_shift": {"max_shift": 25, "p": 0.5},
                "channel_dropout": {"p_drop": 0.1, "p": 0.5},
                "time_mask": {"max_mask_length": 50, "n_masks": 2, "p": 0.5},
                "amplitude_scale": {"scale_range": [0.8, 1.2], "p": 0.5},
            }
    
    Returns:
        Compose transform or None if no augmentations specified
    """
    if aug_config is None or not aug_config:
        return None
    
    transforms = []
    
    if "gaussian_noise" in aug_config:
        cfg = aug_config["gaussian_noise"]
        transforms.append(GaussianNoise(
            std=cfg.get("std", 0.1),
            p=cfg.get("p", 0.5)
        ))
    
    if "time_shift" in aug_config:
        cfg = aug_config["time_shift"]
        transforms.append(TimeShift(
            max_shift=cfg.get("max_shift", 25),
            p=cfg.get("p", 0.5)
        ))
    
    if "channel_dropout" in aug_config:
        cfg = aug_config["channel_dropout"]
        transforms.append(ChannelDropout(
            p_drop=cfg.get("p_drop", 0.1),
            p=cfg.get("p", 0.5)
        ))
    
    if "time_mask" in aug_config:
        cfg = aug_config["time_mask"]
        transforms.append(TimeMask(
            max_mask_length=cfg.get("max_mask_length", 50),
            n_masks=cfg.get("n_masks", 2),
            p=cfg.get("p", 0.5)
        ))
    
    if "amplitude_scale" in aug_config:
        cfg = aug_config["amplitude_scale"]
        scale_range = cfg.get("scale_range", [0.8, 1.2])
        if isinstance(scale_range, list):
            scale_range = tuple(scale_range)
        transforms.append(AmplitudeScale(
            scale_range=scale_range,
            p=cfg.get("p", 0.5)
        ))
    
    if "channel_permute" in aug_config:
        cfg = aug_config["channel_permute"]
        transforms.append(ChannelPermute(
            n_permute=cfg.get("n_permute", 2),
            p=cfg.get("p", 0.3)
        ))
    
    if not transforms:
        return None
    
    return Compose(transforms)

