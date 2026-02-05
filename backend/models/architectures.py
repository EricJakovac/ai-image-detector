import torch.nn as nn
import timm


# Modeli sa fine-tuningom - SVI imaju self.model umjesto self.backbone
class EfficientNetTest(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = timm.create_model(  # 👈 Promijeni backbone -> model
            "efficientnet_b0", pretrained=False, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


class ViTTest(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = timm.create_model(  # 👈 Promijeni backbone -> model
            "vit_base_patch16_224", pretrained=False, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


class DeiTTest(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = timm.create_model(
            "deit_base_patch16_224", pretrained=False, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


class ConvNeXtTest(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = timm.create_model(
            "convnext_tiny", pretrained=False, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


# Raw modeli bez fine-tuninga - TAKOĐER promijeni backbone -> model
class EfficientNetRaw(nn.Module):
    """Raw EfficientNet bez fine-tune (ImageNet weights)"""

    def __init__(self, num_classes=2):
        super().__init__()
        # Pretrained ImageNet model
        self.model = timm.create_model(  # 👈 Promijeni backbone -> model
            "efficientnet_b0",
            pretrained=True,
            num_classes=1000,
        )

        # Freeze backbone - ne treba ga trenirati
        for param in self.model.parameters():
            param.requires_grad = False

        # Dodaj novi classifier za AI/Real
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.model(x)  # 👈 Promijeni
        return self.classifier(features)


class ViTRaw(nn.Module):
    """Raw ViT bez fine-tune (ImageNet weights)"""

    def __init__(self, num_classes=2):
        super().__init__()
        # Pretrained ImageNet ViT
        self.model = timm.create_model(  # 👈 Promijeni backbone -> model
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=1000,
        )

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Simple classifier
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        features = self.model(x)  # 👈 Promijeni
        return self.classifier(features)
