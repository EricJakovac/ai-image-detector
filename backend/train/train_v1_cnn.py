import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from sklearn.model_selection import train_test_split
from datetime import datetime
from PIL import Image, ImageFile
import torch_directml
import gc

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# --- DEFINICIJA MODELA ---
class EfficientNetTest(nn.Module):
    """
    EfficientNet-B0 model za binary klasifikaciju (AI vs real).
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=3,
        )

    def forward(self, x):
        return self.backbone(x)


def train_cnn_windows_native():
    # 1. Postava uređaja (DirectML za AMD)
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"🔥 DETEKTIRANA SNAGA (CNN): {torch_directml.device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU NIJE PRONAĐEN!")

    # 2. Parametri treninga
    MODEL_NAME = "EfficientNet_B0_V2"
    IMG_SIZE = 224
    BATCH_SIZE = 128
    EPOCHS = 5
    LEARNING_RATE = 2e-4

    # 3. Pametno upravljanje putanjama
    script_path = os.path.abspath(__file__)
    train_dir = os.path.dirname(script_path)
    backend_dir = os.path.dirname(train_dir)
    root_dir = os.path.dirname(backend_dir)

    # Putanje do podataka i spremanja
    DATA_DIR = os.path.join(root_dir, "ai_vs_real_84k_train_data")
    SAVE_DIR = os.path.join(backend_dir, "models")

    print(f"------------------------------------------")
    print(f"🚀 POKRETANJE CNN TRENINGA")
    print(f"📂 Dataset: {DATA_DIR}")
    print(f"💾 Spremanje u: {SAVE_DIR}")
    print(f"------------------------------------------")

    if not os.path.exists(DATA_DIR):
        print(f"❌ GREŠKA: Dataset nije pronađen na lokaciji: {DATA_DIR}")
        return

    # 4. Priprema podataka
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = ImageFolder(DATA_DIR, transform=transform)
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, stratify=full_dataset.targets, random_state=42
    )

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 5. Inicijalizacija modela
    model = EfficientNetTest(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Trening petlja
    for epoch in range(EPOCHS):
        model.train()
        print(f"\n📅 CNN Epoha {epoch+1}/{EPOCHS}")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            del outputs

            if batch_idx % 20 == 0:
                gc.collect()
                print(
                    f"⚡ [CNN Train] Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )

            del loss

        # --- VALIDACIJA ---
        model.eval()
        correct, total = 0, 0
        print(f"🧪 Validacija epohe {epoch+1} u tijeku...")
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                del outputs

        acc = 100.0 * correct / total
        print(f"📊 Rezultat epohe {epoch+1}: Preciznost: {acc:.2f}%")

        # --- SPREMANJE ---
        os.makedirs(SAVE_DIR, exist_ok=True)
        acc_str = f"{acc:.2f}"
        # Formatiranje naziva datoteke
        file_name = (
            f"EffNetV2_IMG{IMG_SIZE}_B{BATCH_SIZE}_LR2e-4_Acc{acc_str}_E{epoch+1}.pth"
        )
        epoch_save_path = os.path.join(SAVE_DIR, file_name)

        try:
            model.to("cpu")
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "class_to_idx": full_dataset.class_to_idx,
                "accuracy": f"{acc:.2f}%",
                "epoch": epoch + 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            torch.save(
                checkpoint, epoch_save_path, _use_new_zipfile_serialization=False
            )
            print(f"✅ SPREMLJENO: {file_name}")
            model.to(device)
        except Exception as e:
            print(f"❌ Greška pri spremanju epohe: {e}")
            model.to(device)

    print(f"\n🏁 CNN TRENING KOMPLETAN!")


if __name__ == "__main__":
    train_cnn_windows_native()
