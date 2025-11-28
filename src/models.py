import math

import torch
import torch.nn as nn


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

