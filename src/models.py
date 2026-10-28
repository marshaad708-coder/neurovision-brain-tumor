"""Backbone for NeuroVision: ResNet-18 with a 4-class classification head.

Classes (folder order): 0 = glioma, 1 = meningioma, 2 = notumor, 3 = pituitary.
`feature_map()` exposes the last convolutional map (used by Grad-CAM) and
`features()` the pooled 512-d embedding.
"""
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class NeuroNet(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True,
                 freeze_until: str = "layer2"):
        super().__init__()
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=w)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            net.layer1, net.layer2, net.layer3, net.layer4
        self.pool = net.avgpool
        self.dropout = nn.Dropout(p=0.3)
        self.head = nn.Linear(net.fc.in_features, num_classes)

        # CPU-training shortcut, documented in REPORT.md: the early layers
        # (stem through layer2) stay frozen at their ImageNet weights, the
        # network adapts to MRI scans in layer3/layer4 + head. A GPU run
        # would fine-tune everything.
        _order = ["stem", "layer1", "layer2", "layer3", "layer4"]
        frozen = set(_order[: _order.index(freeze_until) + 1]) if freeze_until else set()
        for name, mod in [("stem", self.stem), ("layer1", self.layer1),
                          ("layer2", self.layer2), ("layer3", self.layer3),
                          ("layer4", self.layer4)]:
            if name in frozen:
                for p in mod.parameters():
                    p.requires_grad_(False)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Dense (B, 512, H', W') map before pooling."""
        return self._backbone(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled 512-d embedding."""
        fmap = self._backbone(x)
        return torch.flatten(self.pool(fmap), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.features(x)))
