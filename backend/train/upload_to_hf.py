from huggingface_hub import HfApi
import os

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("REPO_ID")

MODELS_TO_UPLOAD = {
    "cnn_efficientnet": "/app/models/EfficientNetTest_Res224_B64_E5_20260119.pth",
    "vit_transformer": "/app/models/ViT_Base_Patch16_Res224_B32_E5_20260119.pth",
}

api = HfApi()

print(f"🚀 Upload modela na Hugging Face repozitorij: {REPO_ID}\n")

for folder_name, local_path in MODELS_TO_UPLOAD.items():
    if os.path.exists(local_path):
        print(f"📦 Učitavam {folder_name}...")

        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{folder_name}/model.pth",
            repo_id=REPO_ID,
            token=HF_TOKEN,
        )
        print(f"✅ {folder_name} uspješno učitan!\n")
    else:
        print(f"⚠️ Datoteka nije pronađena: {local_path}. Preskačem...")

print(f"✨ Svi dostupni modeli su učitani na: https://huggingface.co/{REPO_ID}")
