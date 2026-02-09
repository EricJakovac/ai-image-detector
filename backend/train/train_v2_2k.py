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

# --- DEVICE ---
device = torch.device("cpu")

# --- GLOBAL PARAMS ---
IMG_SIZE = 224
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
EPOCHS = 12
LR_CLASSIFIER = 5e-5
EARLY_STOP_PATIENCE = 4

# --- DATASET ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "novi_ai_vs_real_dataset_2k")  # 2k dataset
SAVE_DIR = os.path.join(ROOT_DIR, "backend", "models", "v2_models")
V1_MODELS_DIR = os.path.join(ROOT_DIR, "backend", "models", "v1_models")

os.makedirs(SAVE_DIR, exist_ok=True)

# --- TRANSFORMS ---
transform_common = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = ImageFolder(DATA_DIR, transform=transform_common)
train_idx, val_idx = train_test_split(
    range(len(dataset)), test_size=0.2, stratify=dataset.targets, random_state=42
)
train_loader = DataLoader(
    Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)


# --- UTILITY FUNCTION ---
def train_model(model_name, model, load_path, freeze_strategy="default"):
    print(f"\n🔥 DETEKTIRANA SNAGA ({model_name}): CPU")
    checkpoint = torch.load(load_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # prilagodba za timm modele
    if (
        "backbone" in list(state_dict.keys())[0]
        or "model." in list(state_dict.keys())[0]
    ):
        fixed_state_dict = {
            (
                k[9:]
                if k.startswith("backbone.")
                else k[6:] if k.startswith("model.") else k
            ): v
            for k, v in state_dict.items()
        }
    else:
        fixed_state_dict = state_dict
    model.load_state_dict(fixed_state_dict, strict=False)
    model.to(device)

    # Freeze slojeva prema strategiji
    params_to_update, frozen_params = [], 0
    total_layers = len(list(model.parameters()))
    if freeze_strategy == "vit":
        freeze_layers = int(total_layers * 0.8)
        for i, (name, param) in enumerate(model.named_parameters()):
            if i < freeze_layers:
                param.requires_grad = False
                frozen_params += 1
            else:
                param.requires_grad = True
                params_to_update.append(param)
    elif freeze_strategy == "deit":
        for name, param in model.named_parameters():
            if any(
                k in name
                for k in ["head", "classifier", "norm", "blocks.10", "blocks.11"]
            ):
                param.requires_grad = True
                params_to_update.append(param)
            else:
                param.requires_grad = False
        frozen_params = total_layers - len(params_to_update)
    elif freeze_strategy == "convnext":
        for name, param in model.named_parameters():
            if any(f"stages.{s}" in name for s in [0, 1]):
                param.requires_grad = False
                frozen_params += 1
            else:
                param.requires_grad = True
                params_to_update.append(param)
    elif freeze_strategy == "effnet":
        for name, param in model.named_parameters():
            if (
                "backbone.blocks.5" in name
                or "backbone.blocks.6" in name
                or "classifier" in name
            ):
                param.requires_grad = True
                params_to_update.append(param)
            else:
                param.requires_grad = False
                frozen_params += 1

    print(f"❄️ Zamrznuto slojeva: {frozen_params}/{total_layers}")
    print(f"🔥 Dotreniranje slojeva: {len(params_to_update)}")

    optimizer = optim.AdamW(params_to_update, lr=LR_CLASSIFIER, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    early_stop_counter = 0

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

        # VALIDACIJA
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

        # SAVE BEST MODEL ONLY
        if acc > best_acc:
            best_acc = acc
            early_stop_counter = 0
            best_path = os.path.join(
                SAVE_DIR, f"{model_name}_CPU_IMG{IMG_SIZE}_Acc{acc:.2f}_BEST_2k.pth"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "class_to_idx": dataset.class_to_idx,
                    "accuracy": f"{acc:.2f}%",
                    "epoch": epoch + 1,
                    "v1_source": os.path.basename(load_path),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                best_path,
                _use_new_zipfile_serialization=False,
            )
            print(f"✅ SPREMLJEN NAJBOLJI V2 MODEL: {best_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print("🛑 Rani stop zbog nedostatka poboljšanja")
                break

    print(f"🏁 {model_name} treniranje kompletirano! Najbolja točnost: {best_acc:.2f}%")
    return best_acc


# --- DEFINICIJA MODELA ---
models_to_train = [
    (
        "ViT_PT16_V2",
        timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2),
        os.path.join(V1_MODELS_DIR, "ViT_PT16_IMG224_B8_LR1e-5_Acc96.54_E5.pth"),
        "vit",
    ),
    (
        "DeiT_PT16_V2",
        timm.create_model("deit_base_patch16_224", pretrained=False, num_classes=2),
        os.path.join(V1_MODELS_DIR, "DeiT_PT16_IMG224_B8_LR1e-5_Acc95.02_E4.pth"),
        "deit",
    ),
    (
        "ConvNeXt_T_V2",
        timm.create_model("convnext_tiny", pretrained=True, num_classes=2),
        os.path.join(V1_MODELS_DIR, "ConvNeXt_T_IMG224_B16_LR2e-4_Acc89.32_E5.pth"),
        "convnext",
    ),
    (
        "EffNetV2",
        timm.create_model("efficientnet_b0", pretrained=True, num_classes=2),
        os.path.join(V1_MODELS_DIR, "EffNet_IMG224_B128_LR2e-4_Acc95.40_E5.pth"),
        "effnet",
    ),
]

if __name__ == "__main__":
    for name, model, v1_path, strategy in models_to_train:
        if not os.path.exists(v1_path):
            print(f"❌ V1 model nije pronađen: {v1_path}, preskačem {name}")
            continue
        train_model(name, model, v1_path, freeze_strategy=strategy)
