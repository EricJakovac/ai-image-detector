import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

# --- PUTANJE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 1. Učitaj .env
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    print("✅ .env datoteka uspješno učitana.")
else:
    print("⚠️ .env datoteka nije pronađena! Provjeri putanju.")

# 2. Dohvati varijable
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("REPO_ID")

if not REPO_ID:
    print("❌ GREŠKA: REPO_ID nije definiran u .env datoteci!")
    exit(1)

api = HfApi()

# --- KONFIGURACIJA MODELA ---
# Eksplicitne putanje za svaki model
MODELS_CONFIG = [
    {
        "hf_folder": "cnn_efficientnet",
        "local_path": os.path.join(
            MODELS_DIR, "EffNet_IMG224_B128_LR2e-4_Acc95.40_E5.pth"
        ),
        "description": "CNN Fine-tuned (224x224)",
    },
    {
        "hf_folder": "vit_transformer",
        "local_path": os.path.join(
            MODELS_DIR, "ViT_PT16_IMG224_B8_LR1e-5_Acc96.54_E5.pth"
        ),
        "description": "ViT Fine-tuned (224x224)",
    },
    {
        "hf_folder": "cnn_raw",
        "local_path": os.path.join(MODELS_DIR, "cnn_raw", "model.pth"),
        "description": "CNN Raw (ImageNet)",
    },
    {
        "hf_folder": "vit_raw",
        "local_path": os.path.join(MODELS_DIR, "vit_raw", "model.pth"),
        "description": "ViT Raw (ImageNet)",
    },
    {
        "hf_folder": "deit_transformer",
        "local_path": os.path.join(
            MODELS_DIR, "DeiT_PT16_IMG224_B8_LR1e-5_Acc95.02_E4.pth"
        ),
        "description": "DeiT Fine-tuned (224x224)",
    },
    {
        "hf_folder": "convnext_tiny",
        "local_path": os.path.join(
            MODELS_DIR, "ConvNeXt_T_IMG224_B16_LR2e-4_Acc89.32_E5.pth"
        ),
        "description": "ConvNeXt Tiny Fine-tuned (224x224)",
    },
]

print(f"🚀 Pokretanje uploada na Hugging Face: {REPO_ID}")
print(f"📂 Models directory: {MODELS_DIR}\n")

success_count = 0
fail_count = 0
skip_count = 0

for config in MODELS_CONFIG:
    local_path = config["local_path"]
    hf_folder = config["hf_folder"]
    description = config["description"]

    print(f"\n📦 {hf_folder} ({description})")
    print(f"   Local: {local_path}")

    # Provjeri postoji li file
    if not os.path.exists(local_path):
        print(f"   ❌ File not found!")
        fail_count += 1
        continue

    # Provjeri veličinu
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"   Size: {file_size_mb:.1f}MB")

    if file_size_mb < 1:
        print(f"   ⚠️  Skipping - file too small (might be corrupted)")
        skip_count += 1
        continue

    # Upload
    try:
        print(f"   📤 Uploading to HF: {hf_folder}/model.pth")

        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{hf_folder}/model.pth",
            repo_id=REPO_ID,
            token=HF_TOKEN,
        )

        print(f"   ✅ Upload successful")
        success_count += 1

    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        fail_count += 1

# Sažetak
print(f"\n{'='*50}")
print("📊 UPLOAD SUMMARY:")
print(f"   ✅ Success: {success_count}")
print(f"   ❌ Failed:  {fail_count}")
print(f"   ⏭️  Skipped: {skip_count}")

if success_count > 0:
    print(f"\n✨ Models available at: https://huggingface.co/{REPO_ID}")
    print("   Note: It may take a few minutes for models to appear on the website")
else:
    print("\n❌ No models were uploaded successfully")

print(f"\n💡 Tip: To force re-upload, delete the file from HF repository first")
