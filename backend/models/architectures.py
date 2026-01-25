import torch.nn as nn
import timm


# Modeli sa fine-tuningom
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


# Raw modeli bez fine-tuninga
class EfficientNetRaw(nn.Module):
    """Raw EfficientNet bez fine-tune (ImageNet weights)"""

    def __init__(self, num_classes=2):
        super().__init__()
        # Pretrained ImageNet model
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,  # 👈 KLJUČNO: TRUE za RAW!
            num_classes=1000,  # ImageNet ima 1000 klasa
        )

        # Freeze backbone - ne treba ga trenirati
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Dodaj novi classifier za AI/Real
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1000, 256),  # 1000 -> 256
            nn.ReLU(),
            nn.Linear(256, num_classes),  # 256 -> 2
        )

    def forward(self, x):
        features = self.backbone(x)  # ImageNet features
        return self.classifier(features)  # Mapiraj na 2 klase


class ViTRaw(nn.Module):
    """Raw ViT bez fine-tune (ImageNet weights)"""

    def __init__(self, num_classes=2):
        super().__init__()
        # Pretrained ImageNet ViT
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,  # 👈 KLJUČNO: TRUE za RAW!
            num_classes=1000,
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Simple classifier
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
