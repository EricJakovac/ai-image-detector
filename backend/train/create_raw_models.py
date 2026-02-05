import torch
import os

from models.architectures import EfficientNetRaw, ViTRaw


# Kreiraj direktorije ako ne postoje
os.makedirs("models/cnn_raw", exist_ok=True)
os.makedirs("models/vit_raw", exist_ok=True)

print("🔄 Creating RAW models...")

# 1. CNN RAW
print("Creating CNN RAW model...")
cnn_raw = EfficientNetRaw(num_classes=2)
cnn_raw.eval()

# Spremi cijeli model (backbone + classifier)
torch.save(
    {
        "model_state_dict": cnn_raw.state_dict(),
        "model_class": "EfficientNetRaw",
        "config": {
            "backbone": "efficientnet_b0",
            "pretrained": True,
            "num_classes": 2,
            "classifier": "Linear(1000->256->2)",
        },
    },
    "models/cnn_raw/model.pth",
)

print(f"✅ CNN RAW saved: models/cnn_raw/model.pth")
print(f"   Parameters: {sum(p.numel() for p in cnn_raw.parameters()):,}")

# 2. ViT RAW
print("\nCreating ViT RAW model...")
vit_raw = ViTRaw(num_classes=2)
vit_raw.eval()

torch.save(
    {
        "model_state_dict": vit_raw.state_dict(),
        "model_class": "ViTRaw",
        "config": {
            "backbone": "vit_base_patch16_224",
            "pretrained": True,
            "num_classes": 2,
            "classifier": "Linear(1000->2)",
        },
    },
    "models/vit_raw/model.pth",
)

print(f"✅ ViT RAW saved: models/vit_raw/model.pth")
print(f"   Parameters: {sum(p.numel() for p in vit_raw.parameters()):,}")

# 3. Spremi INFO file
info = """# RAW Models Info

Ovi modeli su pretrained na ImageNet i NISU fine-tuned za AI/Real detekciju.

## Modeli:
- EfficientNet-B0 (RAW): Pretrained na ImageNet, classifier head za 2 klase
- ViT-B/16 (RAW): Pretrained na ImageNet, classifier head za 2 klase

## Kako koristiti:
1. Backbone je FROZEN (ImageNet features)
2. Classifier head je RANDOM inicijaliziran
3. Za dobre rezultate, treba LINEAR PROBE na AI/Real podacima

## Datoteke:
- models/cnn_raw/model.pth
- models/vit_raw/model.pth
"""

with open("models/RAW_MODELS_INFO.md", "w") as f:
    f.write(info)

print("\n✨ RAW models created successfully!")
print("📝 Info saved to: models/RAW_MODELS_INFO.md")
