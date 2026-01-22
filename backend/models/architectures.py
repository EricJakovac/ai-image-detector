import torch.nn as nn
import timm


class EfficientNetTest(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)


class ViTTest(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)
