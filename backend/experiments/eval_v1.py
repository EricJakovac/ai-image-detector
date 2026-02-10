import os
import sys
import json
import torch
import random
from tqdm import tqdm
from PIL import ImageFile
from datetime import datetime
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.architectures import EfficientNetTest, ViTTest, DeiTTest, ConvNeXtTest

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "backend", "models", "v1_models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

IDX_TO_CLASS = {0: "Real", 1: "AI"}

# ---------------- MODELI V1 ----------------
MODELS_CONFIG = {
    "effnet_v1": {
        "model_class": EfficientNetTest,
        "checkpoint_path": os.path.join(
            MODELS_DIR, "EffNet_IMG224_B128_LR2e-4_Acc95.40_E5.pth"
        ),
        "input_size": 224,
    },
    "vit_v1": {
        "model_class": ViTTest,
        "checkpoint_path": os.path.join(
            MODELS_DIR, "ViT_PT16_IMG224_B8_LR1e-5_Acc96.54_E5.pth"
        ),
        "input_size": 224,
    },
    "deit_v1": {
        "model_class": DeiTTest,
        "checkpoint_path": os.path.join(
            MODELS_DIR, "DeiT_PT16_IMG224_B8_LR1e-5_Acc95.02_E4.pth"
        ),
        "input_size": 224,
    },
    "convnext_v1": {
        "model_class": ConvNeXtTest,
        "checkpoint_path": os.path.join(
            MODELS_DIR, "ConvNeXt_T_IMG224_B16_LR2e-4_Acc89.32_E5.pth"
        ),
        "input_size": 224,
    },
}


# ---------------- DATASET ----------------
def load_hf_dataset_subset():
    print("📥 Učitavam HuggingFace dataset...")
    ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")

    data = []

    for split in ds.keys():
        for item in ds[split]:
            img = item["image"]
            label = int(item["label"])  # 0=Real, 1=AI
            data.append({"image": img, "label": label})

    total = len(data)

    random.shuffle(data)  # 🔥 bitno da ne uzmemo samo jednu klasu

    tenth = total // 10
    subset = data[:tenth]

    labels = [d["label"] for d in subset]
    num_real = labels.count(0)
    num_ai = labels.count(1)

    print(f"📊 Ukupno slika: {total}")
    print(f"📉 Koristim samo 1/10 dataset-a: {len(subset)} slika")
    print(f"🧮 Real: {num_real} | AI: {num_ai}")

    return subset, num_real, num_ai


def load_model(model_config):
    model = model_config["model_class"](num_classes=2)
    checkpoint_path = model_config["checkpoint_path"]

    device = torch.device("cpu")
    print(f"🔍 Učitavam model: {os.path.basename(checkpoint_path)}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint nije pronađen: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    fixed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = key
        else:
            new_key = "model." + key
        fixed_state_dict[new_key] = value

    try:
        model.load_state_dict(fixed_state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(fixed_state_dict, strict=False)

    model.to(device)
    model.eval()
    return model, device


def predict_batch(model, images, transform, device, batch_size=8):
    predictions = []
    probabilities = []

    for i in tqdm(range(0, len(images), batch_size), desc="Predviđanje"):
        batch = images[i : i + batch_size]
        batch_imgs = [transform(img.convert("RGB")) for img in batch]
        batch_tensor = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    return predictions, probabilities


def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(
        precision_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    metrics["recall"] = float(
        recall_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    metrics["f1_score"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    y_prob_ai = [float(p[1]) for p in y_prob]

    if len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob_ai))
    else:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = float(tn / (tn + fp) if (tn + fp) > 0 else 0)
        metrics["sensitivity"] = float(tp / (tp + fn) if (tp + fn) > 0 else 0)
    else:
        metrics["specificity"] = 0.0
        metrics["sensitivity"] = 0.0

    return metrics


def evaluate_model(model_name, model_config, eval_data, num_real, num_ai):
    print(f"\n📊 Evaluacija modela: {model_name.upper()}")

    transform = transforms.Compose(
        [
            transforms.Resize((model_config["input_size"], model_config["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    model, device = load_model(model_config)

    images = [d["image"] for d in eval_data]
    y_true = [d["label"] for d in eval_data]

    y_pred, y_prob = predict_batch(model, images, transform, device)

    metrics = calculate_metrics(y_true, y_pred, y_prob)
    metrics["num_samples"] = len(eval_data)
    metrics["num_real"] = num_real
    metrics["num_ai"] = num_ai

    print(
        f"📊 {model_name}: Acc {metrics['accuracy']:.4f}, "
        f"F1 {metrics['f1_score']:.4f}, ROC AUC {metrics['roc_auc']}"
    )

    return metrics


def main():
    print("🚀 EVALUACIJA V1 MODELA (1/10 HF DATASET)")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    eval_data, num_real, num_ai = load_hf_dataset_subset()

    all_results = {}

    for model_name, model_config in MODELS_CONFIG.items():
        try:
            metrics = evaluate_model(
                model_name, model_config, eval_data, num_real, num_ai
            )
            all_results[model_name] = metrics
        except Exception as e:
            print(f"❌ Greška pri evaluaciji {model_name}: {e}")
            all_results[model_name] = {
                "error": str(e),
                "num_samples": 0,
                "num_real": num_real,
                "num_ai": num_ai,
            }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"eval_v1_results_{timestamp}.json")

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Rezultati spremljeni u: {results_file}")
    print("🏁 Evaluacija završena!")


if __name__ == "__main__":
    main()
