# backend/test_raw_models.py
import torch
from torchvision import transforms
from PIL import Image
from models.architectures import EfficientNetRaw, ViTRaw
import numpy as np

# Učitaj modele
print("🧪 Testing RAW models...")

cnn_raw = EfficientNetRaw(num_classes=2)
vit_raw = ViTRaw(num_classes=2)

# Testna slika (crni kvadrat)
test_image = Image.new("RGB", (224, 224), color="black")

# Transformacija
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

input_tensor = transform(test_image).unsqueeze(0)

# Testiraj
with torch.no_grad():
    cnn_out = cnn_raw(input_tensor)
    vit_out = vit_raw(input_tensor)

    cnn_probs = torch.softmax(cnn_out, dim=1)[0]
    vit_probs = torch.softmax(vit_out, dim=1)[0]

    print(f"CNN RAW probs: AI={cnn_probs[0]:.4f}, Real={cnn_probs[1]:.4f}")
    print(f"ViT RAW probs: AI={vit_probs[0]:.4f}, Real={vit_probs[1]:.4f}")

    # RAW modeli bi trebali dati ~0.5/0.5 (nisu sigurni)
    print("\n🎯 Expected: ~0.5/0.5 (random guessing)")
    print("   If probs are ~0.5/0.5: RAW models working correctly")
    print("   If one class >> 0.5: classifier head biased")
