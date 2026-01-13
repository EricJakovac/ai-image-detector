import torch
from app.models.efficientnet_test import EfficientNetTest

def print_model_layers(model):
    """Ispiši sve Conv2d layere."""
    print("=== SVI Conv2d layere u modelu ===")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"  {name}: {type(module).__name__} (out_channels={module.out_channels})")

# Učitaj model
model = EfficientNetTest(num_classes=2)
model.load_trained_weights("models/efficientnet_test_mini.pth")

print_model_layers(model)

# 🎯 Pronađi KANDIDATE za zadnji layer
print("\n=== ZADNJI 10 Conv2d layere (najvjerojatniji za Grad-CAM) ===")
conv_layers = [(name, module) for name, module in model.named_modules() 
               if isinstance(module, torch.nn.Conv2d)]
for name, module in conv_layers[-10:]:
    print(f"  🎯 {name}")
