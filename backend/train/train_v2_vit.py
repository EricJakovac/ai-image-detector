import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from datetime import datetime
import timm
import gc
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_vit_v2_1k():
    # --- DEVICE ---
    device = torch.device("cpu")
    print(f"🔥 DETEKTIRANA SNAGA (ViT V2): CPU")

    # --- PARAMETRI ---
    MODEL_NAME = "ViT_PT16_V2"
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
        "ViT_PT16_IMG224_B8_LR1e-5_Acc96.54_E5.pth",
    )
    SAVE_DIR = os.path.join(root_dir, "backend", "models", "v2_models")

    # --- TRANSFORMACIJE ---
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
    checkpoint = torch.load(V1_MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    fixed_state_dict = {
        (
            k[9:]
            if k.startswith("backbone.")
            else k[6:] if k.startswith("model.") else k
        ): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(fixed_state_dict, strict=False)
    model.to(device)

    # Freeze 80% slojeva
    total_layers = len(list(model.parameters()))
    freeze_layers = int(total_layers * 0.8)
    params_to_update, frozen_params = [], 0
    for i, (name, param) in enumerate(model.named_parameters()):
        if i < freeze_layers:
            param.requires_grad = False
            frozen_params += 1
        else:
            param.requires_grad = True
            params_to_update.append(param)

    print(f"❄️ Zamrznuto slojeva: {frozen_params}/{total_layers}")
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
            "frozen_layers": frozen_params,
            "trainable_layers": len(params_to_update),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        os.makedirs(SAVE_DIR, exist_ok=True)
        last_path = os.path.join(
            SAVE_DIR,
            f"{MODEL_NAME}_CPU_IMG{IMG_SIZE}_Acc{acc:.2f}_E{epoch+1}_1k.pth",
        )
        torch.save(checkpoint_data, last_path, _use_new_zipfile_serialization=False)

        if acc > best_acc:
            best_acc = acc
            early_stop_counter = 0
            best_path = os.path.join(
                SAVE_DIR,
                f"{MODEL_NAME}_CPU_IMG{IMG_SIZE}_Acc{acc:.2f}_BEST_1k.pth",
            )
            torch.save(checkpoint_data, best_path, _use_new_zipfile_serialization=False)
            print(f"✅ SPREMLJEN NAJBOLJI V2 MODEL: {best_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("🛑 Rani stop zbog nedostatka poboljšanja")
                break

    print(f"🏁 ViT V2 treniranje kompletirano! Najbolja točnost: {best_acc:.2f}%")
    return best_acc


if __name__ == "__main__":
    train_vit_v2_1k()
