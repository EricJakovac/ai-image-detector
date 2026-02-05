import os
import torch
from huggingface_hub import hf_hub_download
from models.architectures import (
    EfficientNetTest,
    ViTTest,
    EfficientNetRaw,
    ViTRaw,
    DeiTTest,
    ConvNeXtTest,
)


def get_model_path_optimized(
    folder_name: str,
    filename: str,
    is_hf_spaces: bool,
    repo_id: str,
    hf_token: str = None,
) -> str:
    """Dohvati putanju do modela (lokalan ili HF)."""
    # Koristite .cache folder unutar models/ umjesto zasebnog model_cache
    hf_cache_path = (
        "/home/user/.cache/huggingface/hub"
        if is_hf_spaces
        else os.path.join("models", ".cache", "huggingface", "hub")
    )

    local_path = os.path.join("models", folder_name, filename)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000000:
        print(f"✔️ Using existing model from local folder: {local_path}")
        return local_path

    print(f"🔍 Checking/Fetching model: {folder_name}/{filename}")

    try:
        # Skini model u HF cache
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{folder_name}/{filename}",
            token=hf_token,
            cache_dir=hf_cache_path,
            local_files_only=False,
        )

        print(f"✅ Model downloaded to cache: {path}")

        # Kopiraj iz cache-a u lokalni models folder
        import shutil

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(path, local_path)
        print(f"📁 Model copied to: {local_path}")

        return local_path

    except Exception as e:
        try:
            # Pokušaj koristiti lokalni cache
            return hf_hub_download(
                repo_id=repo_id,
                filename=f"{folder_name}/{filename}",
                cache_dir=hf_cache_path,
                local_files_only=True,
            )
        except:
            print(f"❌ CRITICAL: Model not found in cache or hub: {e}")
            raise e


def load_cnn_fine_tuned(model_path: str, device: torch.device):
    """Učitaj fine-tuned CNN model."""
    model = EfficientNetTest(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    print(f"🔍 CNN checkpoint keys: {list(checkpoint.keys())}")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"📦 Using 'model_state_dict' from checkpoint")
    else:
        state_dict = checkpoint
        print(f"📦 Using direct state_dict from checkpoint")

    print(f"🔑 CNN state_dict sample keys: {list(state_dict.keys())[:3]}")

    # ISPRAVLJENO: Ukloni dupli prefiks "model.backbone." -> "model."
    fixed_state_dict = {}
    for key, value in state_dict.items():
        # Ako ima "model.backbone.", ukloni "backbone."
        if key.startswith("model.backbone."):
            new_key = "model." + key[15:]  # Ukloni "backbone." iz sredine
        # Ako ima samo "backbone.", dodaj "model."
        elif key.startswith("backbone."):
            new_key = "model." + key[9:]  # Zamijeni "backbone." sa "model."
        # Ako ima samo "model.", ostavi
        elif key.startswith("model."):
            new_key = key  # Već je dobro
        # Ako nema prefiks, dodaj "model."
        else:
            new_key = "model." + key

        fixed_state_dict[new_key] = value

    print(f"🔧 Fixed sample keys: {list(fixed_state_dict.keys())[:3]}")

    # Učitaj sa strict=False za sada
    missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)
    if missing:
        print(f"⚠️ CNN Missing keys: {len(missing)}")
    if unexpected:
        print(f"⚠️ CNN Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model


def load_vit_fine_tuned(model_path: str, device: torch.device):
    """Učitaj fine-tuned ViT model."""
    model = ViTTest(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    print(f"🔍 ViT checkpoint keys: {list(checkpoint.keys())}")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"📦 Using 'model_state_dict' from checkpoint")
    else:
        state_dict = checkpoint
        print(f"📦 Using direct state_dict from checkpoint")

    print(f"🔑 ViT state_dict sample keys: {list(state_dict.keys())[:3]}")

    # ISPRAVLJENO: Ista logika za ViT
    fixed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model.backbone."):
            new_key = "model." + key[15:]
        elif key.startswith("backbone."):
            new_key = "model." + key[9:]
        elif key.startswith("model."):
            new_key = key
        else:
            new_key = "model." + key

        fixed_state_dict[new_key] = value

    print(f"🔧 ViT fixed sample keys: {list(fixed_state_dict.keys())[:3]}")

    missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)
    if missing:
        print(f"⚠️ ViT Missing keys: {len(missing)}")
    if unexpected:
        print(f"⚠️ ViT Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model


def load_cnn_raw(model_path: str, device: torch.device):
    """Učitaj raw CNN model."""
    model = EfficientNetRaw(num_classes=2)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        print(f"🔍 CNN Raw checkpoint keys: {list(checkpoint.keys())}")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        print(f"🔑 CNN Raw state_dict sample keys: {list(state_dict.keys())[:3]}")

        # ISPRAVLJENO: Za raw modele
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model.backbone."):
                new_key = "model." + key[15:]
            elif key.startswith("backbone."):
                new_key = "model." + key[9:]
            elif key.startswith("model."):
                new_key = key
            else:
                new_key = "model." + key

            fixed_state_dict[new_key] = value

        model.load_state_dict(fixed_state_dict, strict=False)
        print("✅ CNN Raw loaded with fixed keys")
    except Exception as e:
        print(f"⚠️ CNN Raw loading failed: {e}")
        print("⚠️ Using untrained weights for CNN Raw")

    model.to(device)
    model.eval()
    return model


def load_vit_raw(model_path: str, device: torch.device):
    """Učitaj raw ViT model."""
    model = ViTRaw(num_classes=2)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        print(f"🔍 ViT Raw checkpoint keys: {list(checkpoint.keys())}")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        print(f"🔑 ViT Raw state_dict sample keys: {list(state_dict.keys())[:3]}")

        # ISPRAVLJENO: Za raw modele
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model.backbone."):
                new_key = "model." + key[15:]
            elif key.startswith("backbone."):
                new_key = "model." + key[9:]
            elif key.startswith("model."):
                new_key = key
            else:
                new_key = "model." + key

            fixed_state_dict[new_key] = value

        model.load_state_dict(fixed_state_dict, strict=False)
        print("✅ ViT Raw loaded with fixed keys")
    except Exception as e:
        print(f"⚠️ ViT Raw loading failed: {e}")
        print("⚠️ Using untrained weights for ViT Raw")

    model.to(device)
    model.eval()
    return model


def load_deit_fine_tuned(model_path: str, device: torch.device):
    """Učitaj fine-tuned DeiT model."""
    model = DeiTTest(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    print(f"🔍 DeiT checkpoint keys: {list(checkpoint.keys())}")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"📦 Using 'model_state_dict' from checkpoint")
    else:
        state_dict = checkpoint
        print(f"📦 Using direct state_dict from checkpoint")

    print(f"🔑 DeiT state_dict sample keys: {list(state_dict.keys())[:3]}")

    # ISPRAVLJENO: DeiT - isti problem
    fixed_state_dict = {}
    for key, value in state_dict.items():
        # DeiT možda ima drugačije prefikse
        if key.startswith("model.backbone."):
            new_key = "model." + key[15:]
        elif key.startswith("backbone."):
            new_key = "model." + key[9:]
        elif key.startswith("model."):
            new_key = key
        else:
            new_key = "model." + key

        fixed_state_dict[new_key] = value

    print(f"🔧 DeiT fixed sample keys: {list(fixed_state_dict.keys())[:3]}")

    missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)
    if missing:
        print(f"⚠️ DeiT Missing keys: {len(missing)}")
    if unexpected:
        print(f"⚠️ DeiT Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model


def load_convnext_fine_tuned(model_path: str, device: torch.device):
    """Učitaj fine-tuned ConvNeXt model."""
    model = ConvNeXtTest(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    print(f"🔍 ConvNeXt checkpoint keys: {list(checkpoint.keys())}")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"📦 Using 'model_state_dict' from checkpoint")
    else:
        state_dict = checkpoint
        print(f"📦 Using direct state_dict from checkpoint")

    print(f"🔑 ConvNeXt state_dict sample keys: {list(state_dict.keys())[:3]}")

    # ISPRAVLJENO: ConvNeXt - isti problem
    fixed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model.backbone."):
            new_key = "model." + key[15:]
        elif key.startswith("backbone."):
            new_key = "model." + key[9:]
        elif key.startswith("model."):
            new_key = key
        else:
            new_key = "model." + key

        fixed_state_dict[new_key] = value

    print(f"🔧 ConvNeXt fixed sample keys: {list(fixed_state_dict.keys())[:3]}")

    missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)
    if missing:
        print(f"⚠️ ConvNeXt Missing keys: {len(missing)}")
    if unexpected:
        print(f"⚠️ ConvNeXt Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model


def initialize_all_models(
    device: torch.device, is_hf_spaces: bool, repo_id: str, hf_token: str = None
):
    """Inicijaliziraj SVE modele (stare + nove)."""
    models = {}

    try:
        # --- POSTOJEĆI MODELI ---
        print("\n📦 Loading FINE-TUNED models...")

        # CNN Fine-tuned
        cnn_path = get_model_path_optimized(
            "cnn_efficientnet", "model.pth", is_hf_spaces, repo_id, hf_token
        )
        models["cnn_fine_tuned"] = load_cnn_fine_tuned(cnn_path, device)
        print("✅ CNN Fine-tuned loaded")

        # ViT Fine-tuned
        vit_path = get_model_path_optimized(
            "vit_transformer", "model.pth", is_hf_spaces, repo_id, hf_token
        )
        models["vit_fine_tuned"] = load_vit_fine_tuned(vit_path, device)
        print("✅ ViT Fine-tuned loaded")

        # DeiT Fine-tuned
        try:
            deit_path = get_model_path_optimized(
                "deit_transformer", "model.pth", is_hf_spaces, repo_id, hf_token
            )
            models["deit_fine_tuned"] = load_deit_fine_tuned(deit_path, device)
            print("✅ DeiT Fine-tuned loaded")
        except Exception as e:
            print(f"⚠️ DeiT loading failed: {e}")
            # Možeš preskočiti ili kreirati prazan model
            models["deit_fine_tuned"] = None

        # ConvNeXt Fine-tuned
        try:
            convnext_path = get_model_path_optimized(
                "convnext_tiny", "model.pth", is_hf_spaces, repo_id, hf_token
            )
            models["convnext_fine_tuned"] = load_convnext_fine_tuned(
                convnext_path, device
            )
            print("✅ ConvNeXt Fine-tuned loaded")
        except Exception as e:
            print(f"⚠️ ConvNeXt loading failed: {e}")
            models["convnext_fine_tuned"] = None

        # --- RAW MODELI (samo CNN i ViT kao dosad) ---
        print("\n📦 Loading RAW models...")

        # CNN Raw
        try:
            cnn_raw_path = get_model_path_optimized(
                "cnn_raw", "model.pth", is_hf_spaces, repo_id, hf_token
            )
            models["cnn_raw"] = load_cnn_raw(cnn_raw_path, device)
            print("✅ CNN Raw loaded")
        except Exception as e:
            print(f"⚠️ CNN Raw loading failed: {e}")
            models["cnn_raw"] = None

        # ViT Raw
        try:
            vit_raw_path = get_model_path_optimized(
                "vit_raw", "model.pth", is_hf_spaces, repo_id, hf_token
            )
            models["vit_raw"] = load_vit_raw(vit_raw_path, device)
            print("✅ ViT Raw loaded")
        except Exception as e:
            print(f"⚠️ ViT Raw loading failed: {e}")
            models["vit_raw"] = None

        print(
            f"\n✨ {len([m for m in models.values() if m is not None])} models initialized successfully"
        )

    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        import traceback

        traceback.print_exc()
        raise

    return models
