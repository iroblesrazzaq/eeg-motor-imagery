import inspect
import math

import torch
import torch.nn as nn

try:
    from braindecode.models import ATCNet as BraindecodeATCNet
except ImportError:  # braindecode is optional unless ATCNet is requested
    BraindecodeATCNet = None


class EEGNet(nn.Module):
    """
    Minimal EEGNet-style architecture (inspired by Lawhern et al. 2018).

    Expects input of shape [B, 1, C, T].
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        kernel_length_sep: int = 16,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise_conv = nn.Conv2d(
            F1,
            F1 * D,
            (n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)

        self.separable_depth = nn.Conv2d(
            F1 * D,
            F1 * D,
            (1, kernel_length_sep),
            padding=(0, kernel_length_sep // 2),
            groups=F1 * D,
            bias=False,
        )
        self.separable_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(dropout)

        feat_dim = self._compute_feature_dim(n_samples, pool1=pool1, pool2=pool2)
        self.classifier = nn.Linear(F2 * feat_dim, n_classes)
        self.activation = nn.ELU()

    def _compute_feature_dim(self, n_samples: int, pool1: int, pool2: int) -> int:
        # Conv padding keeps temporal length; only pooling reduces it.
        out = math.floor(n_samples / pool1)
        out = math.floor(out / pool2)
        if out <= 0:
            raise ValueError("Pooling too aggressive for given n_samples.")
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, 1, C, T]
        Returns:
            logits: Tensor of shape [B, n_classes]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_eegnet_from_config(config: dict, n_channels: int, n_samples: int) -> EEGNet:
    model_cfg = config.get("model", {})
    return EEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=model_cfg.get("n_classes", 2),
        dropout=model_cfg.get("dropout", 0.5),
        F1=model_cfg.get("F1", 8),
        D=model_cfg.get("D", 2),
        F2=model_cfg.get("F2", 16),
        kernel_length=model_cfg.get("kernel_length", 64),
        kernel_length_sep=model_cfg.get("kernel_length_sep", 16),
        pool1=model_cfg.get("pool1", 4),
        pool2=model_cfg.get("pool2", 8),
    )


def _filter_kwargs_for_signature(kwargs: dict, signature: inspect.Signature):
    filtered = {}
    unused = {}
    for key, value in kwargs.items():
        if key in signature.parameters:
            filtered[key] = value
        else:
            unused[key] = value
    return filtered, unused


def build_atcnet_from_config(config: dict, n_channels: int, n_samples: int):
    """
    Build ATCNet from Braindecode with a config-driven interface.
    """
    if BraindecodeATCNet is None:
        raise ImportError("braindecode is required for ATCNet. Install with `pip install braindecode`.")

    model_cfg = config.get("model", {})
    sfreq = model_cfg.get("sfreq")
    if sfreq is None:
        raise ValueError("model.sfreq must be set for ATCNet (sampling frequency in Hz).")

    # Required arguments
    base_kwargs = {
        "n_chans": n_channels,
        "n_outputs": model_cfg.get("n_classes", 2),
        "n_classes": model_cfg.get("n_classes", 2),
        "input_window_samples": n_samples,
        "sfreq": sfreq,
    }

    # Optional arguments are passed through if they exist in the signature
    extra_kwargs = {k: v for k, v in model_cfg.items() if k not in {"name", "n_classes", "sfreq"}}
    candidate_kwargs = {**base_kwargs, **extra_kwargs}

    sig = inspect.signature(BraindecodeATCNet)
    kwargs, unused = _filter_kwargs_for_signature(candidate_kwargs, sig)
    # Map between potential n_outputs/n_classes naming differences
    if "n_outputs" not in kwargs and "n_outputs" in sig.parameters and "n_classes" in candidate_kwargs:
        kwargs["n_outputs"] = candidate_kwargs["n_classes"]
    if "n_classes" not in kwargs and "n_classes" in sig.parameters and "n_outputs" in candidate_kwargs:
        kwargs["n_classes"] = candidate_kwargs["n_outputs"]

    missing = [p for p, param in sig.parameters.items() if param.default is inspect._empty and p not in kwargs]
    if missing:
        raise ValueError(f"ATCNet missing required arguments: {missing}")

    if unused:
        print(f"Warning: unused ATCNet config keys: {sorted(unused.keys())}")

    return BraindecodeATCNet(**kwargs)


def build_model_from_config(config: dict, n_channels: int, n_samples: int):
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "eegnet").lower()
    if name == "eegnet":
        return build_eegnet_from_config(config, n_channels=n_channels, n_samples=n_samples)
    if name == "atcnet":
        return build_atcnet_from_config(config, n_channels=n_channels, n_samples=n_samples)
    raise ValueError(f"Unsupported model name '{name}'. Use 'eegnet' or 'atcnet'.")
