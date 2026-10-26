import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class DropNet(nn.Module):
    """per-frame mobilenet features -> small temporal conv -> went_in / missed / no_drop

    input: uint8-ish float clip [B, T, 3, H, W] in 0..255, RGB
    """

    def __init__(self, n_classes=3, pretrained=True):
        super().__init__()
        w = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=w).features  # -> 576 ch
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.temporal = nn.Sequential(
            nn.Conv1d(576, 160, 3, padding=1), nn.BatchNorm1d(160), nn.ReLU(inplace=True),
            nn.Conv1d(160, 160, 3, padding=2, dilation=2), nn.BatchNorm1d(160), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(320, n_classes))
        self.register_buffer("mean", MEAN * 255)
        self.register_buffer("std", STD * 255)

    def forward(self, x):
        b, t = x.shape[:2]
        x = (x - self.mean) / self.std
        f = self.pool(self.backbone(x.flatten(0, 1))).flatten(1)  # [B*T, 576]
        f = f.view(b, t, -1).transpose(1, 2)  # [B, 576, T]
        f = self.temporal(f)
        f = torch.cat([f.mean(2), f.amax(2)], 1)
        return self.head(f)


OUTPUTS = ["drop_detected", "drop_not_detected"]


class DetectedHead(nn.Module):
    """what ships: [p(drop went in), p(it didn't)]. missed + no_drop just get merged"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        p = self.net(x).softmax(1)
        return torch.stack([p[:, 0], p[:, 1] + p[:, 2]], 1)
