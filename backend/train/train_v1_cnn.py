import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from models.efficientnet_test import EfficientNetTest
from sklearn.model_selection import train_test_split
from datetime import datetime
from PIL import Image, ImageFile
import torch_directml
import gc

# Postavke za robusnije učitavanje slika
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_cnn_windows_native():
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"🔥 DETEKTIRANA SNAGA (CNN): {torch_directml.device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU NIJE PRONAĐEN!")

    # --- PARAMETRI ---
    MODEL_NAME = "EfficientNetTest_V2"
    IMG_SIZE = 224
    BATCH_SIZE = 128  # Povećano na 128 (EfficientNet to može podnijeti)
    EPOCHS = 5
    LEARNING_RATE = 2e-4
    DATA_DIR = "../ai_vs_real_84k_train_data"  # Ispravljena putanja
    SAVE_DIR = "models"

    print(f"------------------------------------------")
    print(f"🚀 POKRETANJE CNN V2 TRENINGA | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"------------------------------------------")

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if not os.path.exists(DATA_DIR):
        print(f"❌ GREŠKA: Dataset nije pronađen na {DATA_DIR}")
        return

    full_dataset = ImageFolder(DATA_DIR, transform=transform)
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, stratify=full_dataset.targets, random_state=42
    )

    # Smanjen num_workers na 4 radi stabilnosti Windowsa (da ne šteka)
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

    model = EfficientNetTest(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

            # Oslobađanje memorije
            del outputs

            if batch_idx % 20 == 0:
                gc.collect()  # Povremeno očisti Python smeće
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
        print(f"📊 CNN Rezultat epohe {epoch+1}: Preciznost: {acc:.2f}%")

        # --- AUTOMATSKO SPREMANJE NAKON SVAKE EPOHE ---
        print(f"💾 Spremanje checkpointa za CNN epohu {epoch+1}...")
        os.makedirs(SAVE_DIR, exist_ok=True)

        # Naziv file-a sadrži datum i broj epohe
        acc_str = f"{acc:.2f}".replace(".", "_")
        epoch_file_name = f"{MODEL_NAME}_IMG{IMG_SIZE}_B{BATCH_SIZE}_LR{LEARNING_RATE}_Acc{acc_str}_E{epoch+1}.pth"
        epoch_save_path = os.path.join(SAVE_DIR, epoch_file_name)

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
            print(f"✅ CNN Epoha {epoch+1} uspješno spremljena!")
            model.to(device)  # Vrati na GPU
        except Exception as e:
            print(f"❌ Greška pri spremanju epohe: {e}")
            model.to(device)

    print(f"\n🏁 CNN TRENING KOMPLETAN!")


if __name__ == "__main__":
    train_cnn_windows_native()
