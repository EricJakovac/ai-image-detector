import os
import io
import torch
import base64
import asyncio
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from models.architectures import EfficientNetTest, ViTTest, EfficientNetRaw, ViTRaw
from gradcam.gradcam_utils import GradCamGenerator, GradCamGeneratorRaw
from gradcam.vit_utils import ViTAttentionGenerator, ViTAttentionGeneratorRaw

# --- INICIJALIZACIJA APLIKACIJE ---
app = FastAPI(title="AI Image Detector Backend", version="2.0.0")
load_dotenv()

# --- GLOBALNE VARIJABLE ---
IS_HF_SPACES = os.environ.get("SPACE_ID") is not None or "hf.space" in os.environ.get(
    "HOSTNAME", ""
)

REPO_ID = os.getenv("REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

if not REPO_ID:
    raise RuntimeError(
        "❌ REPO_ID environment variable is not set. "
        "Set it in HF Spaces Variables or your local .env file."
    )

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
executor = ThreadPoolExecutor(max_workers=4)  # Povećano za 4 modela
processing_status: Dict[str, dict] = {}
prediction_cache: Dict[str, dict] = {}

# --- CORS KONFIGURACIJA ---
if IS_HF_SPACES:
    allow_origins = ["*"]
else:
    allow_origins = [FRONTEND_URL, "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- POMOĆNE FUNKCIJE ---
def get_model_path_optimized(folder_name: str, filename: str) -> str:
    hf_cache_path = (
        "/home/user/.cache/huggingface/hub" if IS_HF_SPACES else "./model_cache"
    )

    local_path = os.path.join("models", folder_name, filename)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000000:
        print(f"✔️ Using existing model from local folder: {local_path}")
        return local_path

    print(f"🔍 Checking/Fetching model: {folder_name}/{filename}")

    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"{folder_name}/{filename}",
            token=HF_TOKEN,
            cache_dir=hf_cache_path,
            local_files_only=False,
        )
        print(f"✅ Model path ready: {path}")
        return path
    except Exception as e:
        try:
            return hf_hub_download(
                repo_id=REPO_ID,
                filename=f"{folder_name}/{filename}",
                cache_dir=hf_cache_path,
                local_files_only=True,
            )
        except:
            print(f"❌ CRITICAL: Model not found in cache or hub: {e}")
            raise e


def get_cache_key(file_contents: bytes) -> str:
    """Generiraj cache key od fajla."""
    file_hash = hashlib.md5(file_contents).hexdigest()
    return f"comparison_{file_hash}"


# --- INICIJALIZACIJA MODELA ---
print(f"🔄 Initializing AI models on device: {device}...")
print(f"📊 Environment: {'HF Spaces' if IS_HF_SPACES else 'Local Development'}")

# Modeli
cnn_fine_tuned = None
vit_fine_tuned = None
cnn_raw = None
vit_raw = None
gradcam_gen = None
vit_attention_gen = None
gradcam_raw_gen = None
vit_attention_raw_gen = None

try:
    # --- 1. FINE-TUNED MODELS ---
    print("\n📦 Loading FINE-TUNED models...")

    # CNN Fine-tuned
    cnn_path = get_model_path_optimized("cnn_efficientnet", "model.pth")
    cnn_fine_tuned = EfficientNetTest(num_classes=2)
    cnn_sd = torch.load(cnn_path, map_location=device, weights_only=False)
    cnn_fine_tuned.load_state_dict(
        cnn_sd["model_state_dict"] if "model_state_dict" in cnn_sd else cnn_sd
    )
    cnn_fine_tuned.to(device)
    cnn_fine_tuned.eval()
    print("✅ CNN Fine-tuned loaded")

    # ViT Fine-tuned
    vit_path = get_model_path_optimized("vit_transformer", "model.pth")
    vit_fine_tuned = ViTTest(num_classes=2)
    vit_sd = torch.load(vit_path, map_location=device, weights_only=False)

    if "model_state_dict" in vit_sd:
        vit_sd = vit_sd["model_state_dict"]

    # Prilagodba ključeva
    if any(
        k.startswith("backbone.") for k in vit_fine_tuned.state_dict().keys()
    ) and not any(k.startswith("backbone.") for k in vit_sd.keys()):
        vit_sd = {"backbone." + k: v for k, v in vit_sd.items()}

    vit_fine_tuned.load_state_dict(vit_sd)
    vit_fine_tuned.to(device)
    vit_fine_tuned.eval()
    print("✅ ViT Fine-tuned loaded")

    # --- 2. RAW MODELS ---
    print("\n📦 Loading RAW models...")

    # CNN Raw
    try:
        cnn_raw_path = get_model_path_optimized("cnn_raw", "model.pth")
        cnn_raw = EfficientNetRaw(num_classes=2)
        cnn_raw_sd = torch.load(cnn_raw_path, map_location=device, weights_only=False)
        cnn_raw.load_state_dict(cnn_raw_sd["model_state_dict"])
        cnn_raw.to(device)
        cnn_raw.eval()
        print("✅ CNN Raw loaded")
    except Exception as e:
        print(f"⚠️ CNN Raw loading failed, creating from scratch: {e}")
        cnn_raw = EfficientNetRaw(num_classes=2)
        cnn_raw.to(device)
        cnn_raw.eval()
        print("✅ CNN Raw created from scratch")

    # ViT Raw
    try:
        vit_raw_path = get_model_path_optimized("vit_raw", "model.pth")
        vit_raw = ViTRaw(num_classes=2)
        vit_raw_sd = torch.load(vit_raw_path, map_location=device, weights_only=False)
        vit_raw.load_state_dict(vit_raw_sd["model_state_dict"])
        vit_raw.to(device)
        vit_raw.eval()
        print("✅ ViT Raw loaded")
    except Exception as e:
        print(f"⚠️ ViT Raw loading failed, creating from scratch: {e}")
        vit_raw = ViTRaw(num_classes=2)
        vit_raw.to(device)
        vit_raw.eval()
        print("✅ ViT Raw created from scratch")

    # --- 3. VIZUALIZACIJSKI GENERATORI ---
    print("\n🎨 Initializing visualization generators...")

    # Grad-CAM samo za fine-tuned CNN
    gradcam_gen = GradCamGenerator(cnn_fine_tuned)
    print("✅ Grad-CAM generator ready (fine-tuned only)")

    try:
        gradcam_raw_gen = GradCamGeneratorRaw(cnn_raw)
        print("✅ Grad-CAM RAW generator ready")
    except Exception as e:
        print(f"⚠️ Could not initialize Grad-CAM RAW: {e}")
        gradcam_raw_gen = None

    # Attention generator samo za fine-tuned ViT
    try:
        vit_inner_model = (
            vit_fine_tuned.backbone
            if hasattr(vit_fine_tuned, "backbone")
            else vit_fine_tuned
        )
        vit_attention_gen = ViTAttentionGenerator(vit_inner_model)
        print("✅ ViT Attention generator ready (fine-tuned only)")
    except Exception as e:
        print(f"⚠️ Could not initialize ViT Attention: {e}")
        vit_attention_gen = None

    # Attention za RAW
    try:
        vit_raw_inner_model = (
            vit_raw.backbone if hasattr(vit_raw, "backbone") else vit_raw
        )
        vit_attention_raw_gen = ViTAttentionGeneratorRaw(vit_raw_inner_model)
        print("✅ ViT Attention RAW generator ready")
    except Exception as e:
        print(f"⚠️ Could not initialize ViT Attention RAW: {e}")
        vit_attention_raw_gen = None

    print("\n✨ All models initialized successfully")
    print(f"   Loaded: 2 Fine-tuned models + 2 Raw models")

except Exception as e:
    print(f"❌ Model initialization error: {e}")
    import traceback

    traceback.print_exc()
    raise


# Transformacija slike
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# --- VALIDACIJA ---
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


def validate_image(file: UploadFile) -> str:
    """Validiraj uploadanu sliku."""
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"File type {file_ext} not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    return file_ext


# --- ASYNC FUNKCIJE ZA PREDIKCIJU ---
async def process_cnn_fine_tuned_prediction(input_tensor: torch.Tensor) -> dict:
    """Async obrada CNN Fine-tuned predikcije."""
    loop = asyncio.get_event_loop()
    with torch.no_grad():
        cnn_out = await loop.run_in_executor(
            executor, lambda: cnn_fine_tuned(input_tensor.to(device))
        )
        cnn_probs = torch.nn.functional.softmax(cnn_out, dim=1)[0]
        cnn_conf, cnn_class = torch.max(cnn_probs, dim=0)

        return {
            "output": cnn_out,
            "probs": cnn_probs,
            "confidence": cnn_conf.item(),
            "class_idx": cnn_class.item(),
            "probabilities": cnn_probs.cpu().tolist(),
        }


async def process_cnn_raw_prediction(input_tensor: torch.Tensor) -> dict:
    """Async obrada CNN Raw predikcije."""
    loop = asyncio.get_event_loop()
    with torch.no_grad():
        cnn_raw_out = await loop.run_in_executor(
            executor, lambda: cnn_raw(input_tensor.to(device))
        )
        cnn_raw_probs = torch.nn.functional.softmax(cnn_raw_out, dim=1)[0]
        cnn_raw_conf, cnn_raw_class = torch.max(cnn_raw_probs, dim=0)

        return {
            "output": cnn_raw_out,
            "probs": cnn_raw_probs,
            "confidence": cnn_raw_conf.item(),
            "class_idx": cnn_raw_class.item(),
            "probabilities": cnn_raw_probs.cpu().tolist(),
        }


async def process_vit_fine_tuned_prediction(input_tensor: torch.Tensor) -> dict:
    """Async obrada ViT Fine-tuned predikcije."""
    loop = asyncio.get_event_loop()
    with torch.no_grad():
        try:
            vit_out = await loop.run_in_executor(
                executor, lambda: vit_fine_tuned(input_tensor.to(device))
            )
            vit_probs = torch.nn.functional.softmax(vit_out, dim=1)[0]
            vit_conf, vit_class = torch.max(vit_probs, dim=0)

            return {
                "output": vit_out,
                "probs": vit_probs,
                "confidence": vit_conf.item(),
                "class_idx": vit_class.item(),
                "probabilities": vit_probs.cpu().tolist(),
                "error": None,
            }
        except Exception as e:
            print(f"ViT fine-tuned prediction error: {e}")
            return {
                "output": None,
                "probs": torch.tensor([0.5, 0.5]),
                "confidence": 0.5,
                "class_idx": 0,
                "probabilities": [0.5, 0.5],
                "error": str(e),
            }


async def process_vit_raw_prediction(input_tensor: torch.Tensor) -> dict:
    """Async obrada ViT Raw predikcije."""
    loop = asyncio.get_event_loop()
    with torch.no_grad():
        try:
            vit_raw_out = await loop.run_in_executor(
                executor, lambda: vit_raw(input_tensor.to(device))
            )
            vit_raw_probs = torch.nn.functional.softmax(vit_raw_out, dim=1)[0]
            vit_raw_conf, vit_raw_class = torch.max(vit_raw_probs, dim=0)

            return {
                "output": vit_raw_out,
                "probs": vit_raw_probs,
                "confidence": vit_raw_conf.item(),
                "class_idx": vit_raw_class.item(),
                "probabilities": vit_raw_probs.cpu().tolist(),
                "error": None,
            }
        except Exception as e:
            print(f"ViT Raw prediction error: {e}")
            return {
                "output": None,
                "probs": torch.tensor([0.5, 0.5]),
                "confidence": 0.5,
                "class_idx": 0,
                "probabilities": [0.5, 0.5],
                "error": str(e),
            }


async def process_gradcam_fine_tuned(pil_image: Image.Image, target_class: int) -> str:
    """Async generiranje Grad-CAM vizualizacije za fine-tuned CNN."""
    loop = asyncio.get_event_loop()
    try:
        if gradcam_gen:
            cnn_cam = await loop.run_in_executor(
                executor, lambda: gradcam_gen.generate_single(pil_image, target_class)
            )
            return cnn_cam
    except Exception as e:
        print(f"⚠️ Grad-CAM error for fine-tuned: {e}")
    return None


async def process_vit_attention_fine_tuned(pil_image: Image.Image) -> str:
    """Async generiranje ViT attention mape za fine-tuned."""
    loop = asyncio.get_event_loop()
    try:
        if vit_attention_gen:
            vit_map = await loop.run_in_executor(
                executor, lambda: vit_attention_gen.generate_attention_map(pil_image)
            )
            return vit_map
    except Exception as e:
        print(f"⚠️ ViT attention error for fine-tuned: {e}")
    return None


async def process_gradcam_raw(pil_image: Image.Image, target_class: int) -> str:
    """Async generiranje Grad-CAM vizualizacije za RAW CNN."""
    loop = asyncio.get_event_loop()
    try:
        if gradcam_raw_gen:
            cnn_cam = await loop.run_in_executor(
                executor,
                lambda: gradcam_raw_gen.generate_single(pil_image, target_class),
            )
            if cnn_cam:  # 👈 Provjeri da nije None
                return cnn_cam
    except Exception as e:
        print(f"⚠️ Grad-CAM error for raw: {e}")
    return None  # 👈 Vrati None, ne fallback


async def process_vit_attention_raw(pil_image: Image.Image) -> str:
    """Async generiranje ViT attention mape za RAW."""
    loop = asyncio.get_event_loop()
    try:
        if vit_attention_raw_gen:
            vit_map = await loop.run_in_executor(
                executor,
                lambda: vit_attention_raw_gen.generate_attention_map(pil_image),
            )
            return vit_map
    except Exception as e:
        print(f"⚠️ ViT attention error for raw: {e}")
    return None


# --- ENDPOINTI ---
@app.get("/debug/environment")
async def debug_environment():
    """Debug endpoint za provjeru okoline i modela."""
    return {
        "environment": {
            "is_hf_spaces": IS_HF_SPACES,
            "space_id": os.environ.get("SPACE_ID"),
            "hostname": os.environ.get("HOSTNAME"),
            "repo_id": REPO_ID,
            "device": str(device),
        },
        "models_loaded": {
            "cnn_fine_tuned": cnn_fine_tuned is not None,
            "vit_fine_tuned": vit_fine_tuned is not None,
            "cnn_raw": cnn_raw is not None,
            "vit_raw": vit_raw is not None,
        },
        "visualization_generators": {
            "gradcam": gradcam_gen is not None,
            "vit_attention": vit_attention_gen is not None,
        },
    }


@app.get("/progress/{request_id}")
async def get_progress(request_id: str):
    """Endpoint za praćenje progresa."""
    if request_id not in processing_status:
        raise HTTPException(404, "Request ID not found")
    return processing_status[request_id]


async def cleanup_status(request_id: str):
    """Čisti status nakon 5 minuta."""
    await asyncio.sleep(300)
    if request_id in processing_status:
        del processing_status[request_id]


# --- GLAVNI ENDPOINT ZA USPOREDBU ---
@app.post("/predict-comparison")
async def predict_comparison(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = None
):
    """Endpoint za usporedbu RAW vs Fine-tuned modela."""
    request_id = (
        f"comp_{time.time()}_{hashlib.md5(file.filename.encode()).hexdigest()[:8]}"
    )
    processing_status[request_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting comparison...",
        "filename": file.filename,
        "start_time": time.time(),
    }

    try:
        # Faza 1: Učitaj sliku
        processing_status[request_id].update(
            {"progress": 10, "message": "Validating image..."}
        )

        file_ext = validate_image(file)
        contents = await file.read()

        # Provjeri cache
        cache_key = get_cache_key(contents)
        if cache_key in prediction_cache:
            processing_status[request_id].update(
                {"status": "complete", "progress": 100, "message": "Served from cache"}
            )
            return prediction_cache[cache_key]

        processing_status[request_id].update(
            {"progress": 20, "message": "Processing image..."}
        )

        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = pil_image.size

        # Transformacija
        input_tensor = transform(pil_image).unsqueeze(0)

        # Faza 2: Pokreni SVA 4 modela paralelno
        processing_status[request_id].update(
            {"progress": 40, "message": "Running all 4 models..."}
        )

        cnn_ft_task = asyncio.create_task(
            process_cnn_fine_tuned_prediction(input_tensor)
        )
        cnn_raw_task = asyncio.create_task(process_cnn_raw_prediction(input_tensor))
        vit_ft_task = asyncio.create_task(
            process_vit_fine_tuned_prediction(input_tensor)
        )
        vit_raw_task = asyncio.create_task(process_vit_raw_prediction(input_tensor))

        # Čekaj sve rezultate
        cnn_ft_result, cnn_raw_result, vit_ft_result, vit_raw_result = (
            await asyncio.gather(cnn_ft_task, cnn_raw_task, vit_ft_task, vit_raw_task)
        )

        # Faza 3: Generiraj vizualizacije za SVE modele
        processing_status[request_id].update(
            {"progress": 70, "message": "Generating visualizations..."}
        )

        # Generiraj vizualizacije za SVA 4 modela paralelno
        gradcam_ft_task = asyncio.create_task(
            process_gradcam_fine_tuned(pil_image, cnn_ft_result["class_idx"])
        )
        gradcam_raw_task = asyncio.create_task(
            process_gradcam_raw(pil_image, cnn_raw_result["class_idx"])
        )
        attention_ft_task = asyncio.create_task(
            process_vit_attention_fine_tuned(pil_image)
        )
        attention_raw_task = asyncio.create_task(process_vit_attention_raw(pil_image))

        # Čekaj sve vizualizacije
        cnn_ft_cam, cnn_raw_cam, vit_ft_map, vit_raw_map = await asyncio.gather(
            gradcam_ft_task,
            gradcam_raw_task,
            attention_ft_task,
            attention_raw_task,
            return_exceptions=True,  # 👈 Ovo je ključno da ne pukne cijeli endpoint
        )

        # Obradi potencijalne greške u vizualizacijama
        if isinstance(cnn_ft_cam, Exception):
            print(f"⚠️ Fine-tuned Grad-CAM failed: {cnn_ft_cam}")
            cnn_ft_cam = None
        if isinstance(cnn_raw_cam, Exception):
            print(f"⚠️ RAW Grad-CAM failed: {cnn_raw_cam}")
            cnn_raw_cam = None
        if isinstance(vit_ft_map, Exception):
            print(f"⚠️ Fine-tuned ViT attention failed: {vit_ft_map}")
            vit_ft_map = None
        if isinstance(vit_raw_map, Exception):
            print(f"⚠️ RAW ViT attention failed: {vit_raw_map}")
            vit_raw_map = None

        # Faza 4: Pripremi rezultate
        processing_status[request_id].update(
            {"progress": 90, "message": "Preparing results..."}
        )

        labels = ["AI", "Real"]

        def prepare_model_result(result, heatmap, model_type, model_name):
            # Pretpostavka: result["probabilities"] = [ai_prob, real_prob]
            ai_prob = result["probabilities"][0]
            real_prob = result["probabilities"][1]
            label = labels[result["class_idx"]]

            model_data = {
                "label": label,
                "probability": round(result["confidence"], 4),
                "confidence_percent": round(result["confidence"] * 100, 1),
                "raw_probabilities": {
                    "ai": round(ai_prob, 4),
                    "real": round(real_prob, 4),
                },
                "type": model_type,
                "name": model_name,
            }

            # Dodaj vizualizaciju samo ako postoji (i za RAW i za fine-tuned)
            if heatmap:
                model_data["visualization"] = heatmap

            return model_data

        # ISPRAVLJENA LOGIKA: Izračunaj improvement - samo ako su predikcije iste
        cnn_same = cnn_ft_result["class_idx"] == cnn_raw_result["class_idx"]
        vit_same = vit_ft_result["class_idx"] == vit_raw_result["class_idx"]

        # Izračunaj improvement samo za iste predikcije
        cnn_improvement = 0
        if cnn_same:
            # Poboljšanje u pouzdanosti kada su predikcije iste
            cnn_improvement = (
                cnn_ft_result["confidence"] - cnn_raw_result["confidence"]
            ) * 100

        vit_improvement = 0
        if vit_same:
            # Poboljšanje u pouzdanosti kada su predikcije iste
            vit_improvement = (
                vit_ft_result["confidence"] - vit_raw_result["confidence"]
            ) * 100

        # Napravi summary tekst
        if cnn_same and vit_same:
            summary = f"Fine-tuning improves CNN by {cnn_improvement:.1f}% and ViT by {vit_improvement:.1f}%"
        elif cnn_same and not vit_same:
            summary = f"Fine-tuning improves CNN by {cnn_improvement:.1f}% and changes ViT prediction"
        elif not cnn_same and vit_same:
            summary = f"Fine-tuning changes CNN prediction and improves ViT by {vit_improvement:.1f}%"
        else:
            summary = "Fine-tuning changes predictions for both models"

        result = {
            "filename": file.filename,
            "file_type": file_ext,
            "image_size": f"{original_size[0]}x{original_size[1]}",
            "processed_size": "224x224",
            "request_id": request_id,
            "processing_time": round(
                time.time() - processing_status[request_id]["start_time"], 2
            ),
            "models": {
                "cnn_fine_tuned": prepare_model_result(
                    cnn_ft_result, cnn_ft_cam, "fine_tuned", "EfficientNet"
                ),
                "cnn_raw": prepare_model_result(
                    cnn_raw_result, cnn_raw_cam, "raw", "EfficientNet (Raw)"
                ),
                "vit_fine_tuned": prepare_model_result(
                    vit_ft_result, vit_ft_map, "fine_tuned", "Vision Transformer"
                ),
                "vit_raw": prepare_model_result(
                    vit_raw_result, vit_raw_map, "raw", "Vision Transformer (Raw)"
                ),
            },
            "comparison": {
                "cnn_improvement": round(cnn_improvement, 1),
                "vit_improvement": round(vit_improvement, 1),
                "cnn_same_prediction": cnn_same,
                "vit_same_prediction": vit_same,
                # Dodaj informacije o promjeni predikcije
                "cnn_prediction_change": (
                    {
                        "from": labels[cnn_raw_result["class_idx"]],
                        "to": labels[cnn_ft_result["class_idx"]],
                    }
                    if not cnn_same
                    else None
                ),
                "vit_prediction_change": (
                    {
                        "from": labels[vit_raw_result["class_idx"]],
                        "to": labels[vit_ft_result["class_idx"]],
                    }
                    if not vit_same
                    else None
                ),
                "best_model": max(
                    [
                        ("cnn_fine_tuned", cnn_ft_result["confidence"]),
                        ("cnn_raw", cnn_raw_result["confidence"]),
                        ("vit_fine_tuned", vit_ft_result["confidence"]),
                        ("vit_raw", vit_raw_result["confidence"]),
                    ],
                    key=lambda x: x[1],
                )[0],
                "summary": summary,
            },
        }

        # Spremi u cache
        prediction_cache[cache_key] = result
        if len(prediction_cache) > 100:
            oldest_key = next(iter(prediction_cache))
            del prediction_cache[oldest_key]

        # Finalni status
        processing_status[request_id].update(
            {"status": "complete", "progress": 100, "message": "Comparison complete"}
        )

        if background_tasks:
            background_tasks.add_task(cleanup_status, request_id)

        return result

    except Exception as e:
        error_msg = str(e)
        if request_id in processing_status:
            processing_status[request_id].update(
                {"status": "error", "message": error_msg}
            )
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """Batch endpoint za više slika (max 3)."""
    if len(files) > 3:
        raise HTTPException(400, "Maximum 3 images allowed per batch")

    results = []
    for file in files:
        try:
            result = await predict_comparison(file)
            results.append(result)
        except Exception as e:
            results.append(
                {"filename": file.filename, "error": str(e), "status": "failed"}
            )

    return {
        "batch_id": f"batch_{int(time.time())}",
        "total_images": len(files),
        "results": results,
    }


@app.get("/")
async def root():
    """Root endpoint - za Hugging Face Spaces health check."""
    return {
        "status": "online",
        "service": "AI Image Detector API",
        "version": "2.0.0",
        "environment": "HF Spaces" if IS_HF_SPACES else "Local",
        "endpoints": {
            "predict_comparison": "/predict-comparison (POST) - Compare RAW vs Fine-tuned models",
            "progress": "/progress/{request_id} (GET)",
            "batch": "/batch-predict (POST)",
            "debug": "/debug/environment (GET)",
        },
        "models_loaded": {
            "cnn_fine_tuned": cnn_fine_tuned is not None,
            "vit_fine_tuned": vit_fine_tuned is not None,
            "cnn_raw": cnn_raw is not None,
            "vit_raw": vit_raw is not None,
        },
        "device": str(device),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True if port == 8000 else False,
        workers=1,
    )
