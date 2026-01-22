import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

# --- PUTANJE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 1. PRVO učitaj .env
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    print("✅ .env datoteka uspješno učitana.")
else:
    print("⚠️ .env datoteka nije pronađena! Provjeri putanju.")

# 2. TEK ONDA dohvati varijable
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("REPO_ID")

# --- KONFIGURACIJA MODELA ---
MODELS_TO_UPLOAD = {
    "cnn_efficientnet": "EffNetV2_IMG224_B128_LR2e-4_Acc95.40_E5.pth",
    "vit_transformer": "ViT_PT16_IMG224_B8_LR1e-5_Acc96.54_E5.pth",
}

api = HfApi()

if not REPO_ID:
    print("❌ GREŠKA: REPO_ID nije definiran u .env datoteci!")
else:
    print(f"🚀 Pokretanje uploada na Hugging Face: {REPO_ID}")
    print(f"📂 Tražim modele u: {MODELS_DIR}\n")

    for folder_name, file_name in MODELS_TO_UPLOAD.items():
        local_path = os.path.join(MODELS_DIR, file_name)

        if os.path.exists(local_path):
            print(f"📦 Priprema za upload: {file_name}...")
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=f"{folder_name}/model.pth",
                    repo_id=REPO_ID,
                    token=HF_TOKEN,
                )
                print(f"✅ {folder_name} uspješno poslan!\n")
            except Exception as e:
                print(f"❌ Greška pri uploadu {folder_name}: {e}")
        else:
            print(f"⚠️ Datoteka NIJE pronađena: {local_path}")

    print(f"✨ Gotovo! Modeli su dostupni na: https://huggingface.co/{REPO_ID}")
