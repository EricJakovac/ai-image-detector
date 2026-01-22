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

# Postavke za robusnije učitavanje slika
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_vit_windows_native():
    # --- POSTAVA UREĐAJA (AMD GPU) ---
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"🔥 DETEKTIRANA SNAGA (ViT): {torch_directml.device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU NIJE PRONAĐEN!")

    # --- PARAMETRI ---
    MODEL_NAME = "ViT_PT16"
    IMG_SIZE = 224
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 8  # Efektivni batch = 64
    EPOCHS = 5
    LEARNING_RATE = 1e-5

    # --- PAMETNO UPRAVLJANJE PUTANJAMA (Kao na CNN-u) ---
    script_path = os.path.abspath(__file__)
    train_dir = os.path.dirname(script_path)
    backend_dir = os.path.dirname(train_dir)
    root_dir = os.path.dirname(backend_dir)

    DATA_DIR = os.path.join(root_dir, "ai_vs_real_84k_train_data")
    SAVE_DIR = os.path.join(backend_dir, "models")

    print(f"------------------------------------------")
    print(f"🚀 POKRETANJE ViT TRENINGA")
    print(f"📂 Dataset: {DATA_DIR}")
    print(f"💾 Spremanje u: {SAVE_DIR}")
    print(f"------------------------------------------")

    if not os.path.exists(DATA_DIR):
        print(f"❌ GREŠKA: Dataset nije pronađen na {DATA_DIR}")
        return

    # --- PRIPREMA PODATAKA ---
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
    train_indices, val_indices = train_test_split(
        indices, test_size=0.1, stratify=full_dataset.targets, random_state=42
    )

    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # --- INICIJALIZACIJA MODELA ---
    print(f"🏗️ Učitavam ViT pre-trained težine...")
    model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)

    # --- TRENING PETLJA ---
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        print(f"\n📅 ViT Epoha {epoch+1}/{EPOCHS}")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            del outputs

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                gc.collect()

            if (batch_idx + 1) % 40 == 0:
                print(
                    f"⚡ [ViT Train] Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item() * ACCUMULATION_STEPS:.4f}"
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

        acc_val = 100.0 * correct / total
        print(f"📊 Rezultat epohe {epoch+1}: Preciznost: {acc_val:.2f}%")

        # --- AUTOMATSKO SPREMANJE ---
        os.makedirs(SAVE_DIR, exist_ok=True)
        acc_str = f"{acc_val:.2f}"
        # Formatiranje naziva datoteke
        file_name = f"{MODEL_NAME}_IMG{IMG_SIZE}_B{BATCH_SIZE}_LR1e-5_Acc{acc_str}_E{epoch+1}.pth"
        epoch_save_path = os.path.join(SAVE_DIR, file_name)

        try:
            model.to("cpu")
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "class_to_idx": full_dataset.class_to_idx,
                "accuracy": f"{acc_val:.2f}%",
                "epoch": epoch + 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            torch.save(
                checkpoint, epoch_save_path, _use_new_zipfile_serialization=False
            )
            print(f"✅ SPREMLJENO: {file_name}")
            model.to(device)
        except Exception as e:
            print(f"❌ Greška pri spremanju: {e}")
            model.to(device)

    print(f"\n🏁 ViT TRENING ZAVRŠEN!")


if __name__ == "__main__":
    train_vit_windows_native()
