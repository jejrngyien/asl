"""
Model definitions: C3D and R(2+1)D (18-layer style).
Both models avoid temporal downsampling to support T>=1 inputs.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


#### C3D
class C3D(nn.Module):
    """
    C3D-like (Tran et al., ICCV 2015) network adapted to avoid temporal pooling in early stages.
    Expects input [B, C, T, H, W], with typical H=W=112.
    """
    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # no temporal pool

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool3d((None, 1, 1)),  # spatial global pool; keep temporal dim
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.features(x)  # [B, 512, T, 1, 1]
        # Average over remaining spatial dims (1x1 already) and temporal mean pool
        x = x.mean(dim=[3, 4])  # [B, 512, T]
        x = x.mean(dim=2)       # temporal average -> [B, 512]
        x = self.dropout(x)
        return self.classifier(x)


### R(2+1)D-18
class R2Plus1DBlock(nn.Module):
    """
    (2+1)D factorized 3D conv block with residual connection.
    A 3x3x3 conv is replaced by: (1x3x3) -> BN/ReLU -> (3x1x1)
    Downsampling occurs spatially with stride=(1,2,2) in the first conv of the block when needed.
    """
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        super().__init__()
        # Spatial conv (no temporal stride)
        self.conv_s = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3),
                                stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        # Temporal conv
        self.conv_t = nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1),
                                stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        # Residual projection if shape changes
        self.downsample = None
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv_s(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv_t(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class R2Plus1D(nn.Module):
    """
    R(2+1)D-18 (Tran et al., CVPR 2018) style network. No temporal downsampling; spatial downsampling only.
    Stem stride is (1,2,2). Global average pool over T, H, W.
    """
    def __init__(self, num_classes: int, in_channels: int = 3, width_mult: float = 1.0, dropout: float = 0.5):
        super().__init__()
        w = lambda c: max(8, int(c * width_mult))
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, w(64), kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(w(64)),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(w(64), w(64), blocks=2, stride=1)
        self.layer2 = self._make_layer(w(64), w(128), blocks=2, stride=2)
        self.layer3 = self._make_layer(w(128), w(256), blocks=2, stride=2)
        self.layer4 = self._make_layer(w(256), w(512), blocks=2, stride=2)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(w(512), num_classes)

    def _make_layer(self, in_planes: int, out_planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [R2Plus1DBlock(in_planes, out_planes, stride=stride)]
        for _ in range(1, blocks):
            layers.append(R2Plus1DBlock(out_planes, out_planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Global average pool over (T, H, W)
        x = x.mean(dim=[2, 3, 4])
        x = self.dropout(x)
        return self.fc(x)



def build_model(name: str, num_classes: int, in_channels: int = 3, **kwargs) -> nn.Module:
    name = name.lower()
    if name in {"c3d"}:
        return C3D(num_classes=num_classes, in_channels=in_channels, **kwargs)
    elif name in {"r2plus1d", "r(2+1)d", "r2p1d"}:
        return R2Plus1D(num_classes=num_classes, in_channels=in_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")
