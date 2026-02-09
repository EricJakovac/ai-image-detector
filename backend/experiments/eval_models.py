import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.architectures import EfficientNetTest, ViTTest, DeiTTest, ConvNeXtTest
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
EVAL_DATASET_PATH = os.path.join(PROJECT_ROOT, "evaluation_dataset")

MODELS_DIR = os.path.join(PROJECT_ROOT, "backend", "models", "v2_models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

CLASSES = ["Real", "AI"]
CLASS_TO_IDX = {"reall": 0, "fakee": 1}
IDX_TO_CLASS = {0: "Real", 1: "AI"}

# --- SVI MODELI NA CPU ---
MODELS_CONFIG = {
    "effnet_v2_2k": {
        "model_class": EfficientNetTest,
        "checkpoint_path": os.path.join(
            MODELS_DIR, "EffNetV2_CPU_IMG224_Acc90.75_BEST_2k.pth"
        ),
        "input_size": 224,
        "force_device": "cpu",
    },
    "vit_v2_2k": {
        "model_class": ViTTest,
        "checkpoint_path": os.path.join(
            MODELS_DIR, "ViT_PT16_V2_CPU_IMG224_Acc96.50_BEST_2k.pth"
        ),
        "input_size": 224,
        "force_device": "cpu",
    },
    "deit_v2_2k": {
        "model_class": DeiTTest,
        "checkpoint_path": os.path.join(
            MODELS_DIR, "DeiT_PT16_V2_CPU_IMG224_Acc95.50_BEST_2k.pth"
        ),
        "input_size": 224,
        "force_device": "cpu",
    },
    "convnext_v2_2k": {
        "model_class": ConvNeXtTest,
        "checkpoint_path": os.path.join(
            MODELS_DIR, "ConvNeXt_T_V2_CPU_IMG224_Acc95.00_BEST_2k.pth"
        ),
        "input_size": 224,
        "force_device": "cpu",
    },
}


def load_image_safely(image_path):
    try:
        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            return Image.new("RGB", (224, 224), color="black")
        img = Image.open(image_path)
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"⚠️  Greška pri učitavanju slike {image_path}: {e}")
        return Image.new("RGB", (224, 224), color="black")


def load_model(model_config, default_device):
    model = model_config["model_class"](num_classes=2)
    checkpoint_path = model_config["checkpoint_path"]

    # Force CPU
    model_device = torch.device("cpu")
    print(f"🔍 Učitavam model: {os.path.basename(checkpoint_path)} na CPU")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint nije pronađen: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Popravi prefikse ključeva ako je potrebno
    fixed_state_dict = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.device.type != "cpu":
            value = value.cpu()
        if key.startswith("model.backbone."):
            new_key = "model." + key[15:]
        elif key.startswith("backbone."):
            new_key = "model." + key[9:]
        elif key.startswith("model."):
            new_key = key
        else:
            new_key = "model." + key
        fixed_state_dict[new_key] = value

    try:
        model.load_state_dict(fixed_state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(fixed_state_dict, strict=False)

    model.to(model_device)
    model.eval()
    return model, model_device


def load_evaluation_dataset():
    data = []
    labels_path = os.path.join(EVAL_DATASET_PATH, "labels.csv")

    if not os.path.exists(labels_path):
        print("❌ Ne postoji labels.csv!")
        return data

    df = pd.read_csv(labels_path)

    for idx, row in df.iterrows():
        rel_path = str(row["filename"]).strip()  # npr: fakee\Fake_00026.png
        folder_label = str(row["label"]).strip().lower()  # fakee ili reall

        if folder_label not in CLASS_TO_IDX:
            continue

        label_idx = CLASS_TO_IDX[folder_label]
        display_label = IDX_TO_CLASS[label_idx]

        # NORMALIZIRAJ SLASHOVE
        rel_path = rel_path.replace("\\", os.sep).replace("/", os.sep)

        img_path = os.path.join(EVAL_DATASET_PATH, rel_path)

        if os.path.exists(img_path):
            data.append(
                {
                    "image_path": img_path,
                    "label": label_idx,
                    "label_str": display_label,
                    "filename": os.path.basename(img_path),
                }
            )
        else:
            print(f"⚠️ Ne postoji: {img_path}")

    return data


def predict_batch(model, image_paths, transform, device, batch_size=8):
    predictions = []
    probabilities = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Predviđanje"):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = [transform(load_image_safely(p)) for p in batch_paths]
        if not batch_images:
            continue
        batch_tensor = torch.stack(batch_images).to(device)
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
    y_prob_ai = [float(prob[1]) for prob in y_prob]
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob_ai))
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


def evaluate_model(model_name, model_config):
    print(f"\n📊 Evaluacija modela: {model_name.upper()}")
    eval_data = load_evaluation_dataset()
    if not eval_data:
        print("❌ Nema podataka za evaluaciju!")
        return {"error": "Nema evaluacijskih podataka", "num_samples": 0}

    transform = transforms.Compose(
        [
            transforms.Resize((model_config["input_size"], model_config["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    model, device = load_model(model_config, torch.device("cpu"))
    image_paths = [d["image_path"] for d in eval_data]
    y_true = [d["label"] for d in eval_data]
    y_pred, y_prob = predict_batch(model, image_paths, transform, device)

    metrics = calculate_metrics(y_true, y_pred, y_prob)
    metrics["num_samples"] = len(eval_data)
    print(
        f"📊 Rezultati za {model_name}: Accuracy {metrics['accuracy']:.4f}, F1 {metrics['f1_score']:.4f}, ROC AUC {metrics['roc_auc']:.4f}"
    )
    return metrics


def main():
    print("🚀 EVALUACIJA V2 MODELA")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}
    for model_name, model_config in MODELS_CONFIG.items():
        try:
            metrics = evaluate_model(model_name, model_config)
            all_results[model_name] = metrics
        except Exception as e:
            print(f"❌ Greška pri evaluaciji {model_name}: {e}")
            all_results[model_name] = {"error": str(e), "num_samples": 0}

    # Spremi rezultate u JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"eval_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n💾 Rezultati spremljeni u: {results_file}")
    print("🏁 Evaluacija završena!")


if __name__ == "__main__":
    main()
