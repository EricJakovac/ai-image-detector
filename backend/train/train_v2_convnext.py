import os
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
import gc

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_convnext_v2_1k():
    # --- DEVICE ---
    device = torch.device("cpu")
    print("🔥 DETEKTIRANA SNAGA (ConvNeXt V2): CPU")

    # --- PARAMETRI ---
    MODEL_NAME = "ConvNeXt_T_V2"
    IMG_SIZE = 224
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 4
    EPOCHS = 12
    LR_CLASSIFIER = 5e-5

    # --- PUTANJE ---
    script_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    DATA_DIR = os.path.join(root_dir, "novi_ai_vs_real_dataset_1k")  # <--- 1k dataset
    V1_MODEL_PATH = os.path.join(
        root_dir,
        "backend",
        "models",
        "v1_models",
        "ConvNeXt_T_IMG224_B16_LR2e-4_Acc89.32_E5.pth",
    )
    SAVE_DIR = os.path.join(root_dir, "backend", "models", "v2_models")

    # --- TRANSFORMACIJE ---
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, shear=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageFolder(DATA_DIR, transform=transform)
    train_idx, val_idx = train_test_split(
        range(len(dataset)), test_size=0.2, stratify=dataset.targets, random_state=42
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    # --- MODEL ---
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=2)
    model.to(device)

    # --- Load V1 weights (zadnja 2 stage-a + head) ---
    if os.path.exists(V1_MODEL_PATH):
        checkpoint = torch.load(V1_MODEL_PATH, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
        fixed_state_dict = {
            k: v
            for k, v in state_dict.items()
            if any(s in k for s in ["stages.2", "stages.3", "head"])
        }
        model.load_state_dict(fixed_state_dict, strict=False)

    # --- Freeze layers (prva 2 stage-a) ---
    params_to_update, frozen_params = [], 0
    for name, param in model.named_parameters():
        if any(f"stages.{s}" in name for s in [0, 1]):
            param.requires_grad = False
            frozen_params += 1
        else:
            param.requires_grad = True
            params_to_update.append(param)

    print(f"❄️ Zamrznuto slojeva: {frozen_params}/{len(list(model.named_parameters()))}")
    print(f"🔥 Dotreniranje slojeva: {len(params_to_update)}")

    optimizer = optim.AdamW(params_to_update, lr=LR_CLASSIFIER, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc, early_stop_counter, early_stop_patience = 0, 0, 4

    for epoch in range(EPOCHS):
        model.train()
        total_loss, accumulation_loss = 0, 0
        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels) / ACCUMULATION_STEPS
            loss.backward()
            accumulation_loss += loss.item() * ACCUMULATION_STEPS

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                gc.collect()
                total_loss += accumulation_loss
                accumulation_loss = 0

            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(
                    f"⚡ [Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} | Avg Loss: {avg_loss:.4f}"
                )

        # --- VALIDACIJA ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100 * correct / total
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(
            f"📊 Epoha {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Acc {acc:.2f}%"
        )

        # --- SPREMANJE CHECKPOINTA ---
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "class_to_idx": dataset.class_to_idx,
            "accuracy": f"{acc:.2f}%",
            "epoch": epoch + 1,
            "v1_source": os.path.basename(V1_MODEL_PATH),
        }

        os.makedirs(SAVE_DIR, exist_ok=True)
        last_path = os.path.join(
            SAVE_DIR, f"{MODEL_NAME}_CPU_IMG{IMG_SIZE}_Acc{acc:.2f}_E{epoch+1}_1k.pth"
        )
        torch.save(checkpoint_data, last_path, _use_new_zipfile_serialization=False)

        if acc > best_acc:
            best_acc = acc
            early_stop_counter = 0
            best_path = os.path.join(
                SAVE_DIR, f"{MODEL_NAME}_CPU_IMG{IMG_SIZE}_Acc{acc:.2f}_BEST_1k.pth"
            )
            torch.save(checkpoint_data, best_path, _use_new_zipfile_serialization=False)
            print(f"✅ SPREMLJEN NAJBOLJI V2 MODEL: {best_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("🛑 Rani stop zbog nedostatka poboljšanja")
                break

    print(
        f"🏁 ConvNeXt V2 CPU treniranje kompletirano! Najbolja točnost: {best_acc:.2f}%"
    )
    return best_acc


if __name__ == "__main__":
    train_convnext_v2_1k()
