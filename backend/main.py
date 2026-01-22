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
from models.architectures import EfficientNetTest, ViTTest
from gradcam.gradcam_utils import GradCamGenerator
from gradcam.vit_utils import ViTAttentionGenerator

# --- INICIJALIZACIJA APLIKACIJE ---
app = FastAPI(title="AI Image Detector Backend", version="1.0.0")
load_dotenv()

# --- GLOBALNE VARIJABLE ---
REPO_ID = os.getenv("REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
device = torch.device("cpu")
executor = ThreadPoolExecutor(max_workers=2)
processing_status: Dict[str, dict] = {}
prediction_cache: Dict[str, dict] = {}

# --- CORS KONFIGURACIJA ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- POMOĆNE FUNKCIJE ---
def get_model_path(folder_name: str, filename: str) -> str:
    """Dohvati putanju do modela, preuzmi ako ne postoji lokalno."""
    local_dir = os.path.join("models", folder_name)
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)

    if not os.path.exists(local_path):
        print(f"📥 Downloading model: {folder_name}/{filename}")
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=f"{folder_name}/{filename}",
                token=HF_TOKEN,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            return path
        except Exception as e:
            print(f"❌ Download error: {e}")
            raise
    return local_path


def get_cache_key(file_contents: bytes, model_name: str) -> str:
    """Generiraj cache key od fajla."""
    file_hash = hashlib.md5(file_contents).hexdigest()
    return f"{model_name}_{file_hash}"


# --- INICIJALIZACIJA MODELA ---
print("🔄 Initializing AI models...")

# CNN (EfficientNet) model
cnn_model = None
vit_model = None
gradcam_gen = None
vit_attention_gen = None

try:
    # CNN model
    cnn_path = get_model_path("cnn_efficientnet", "model.pth")
    cnn_model = EfficientNetTest(num_classes=2)
    cnn_sd = torch.load(cnn_path, map_location=device, weights_only=True)
    cnn_model.load_state_dict(
        cnn_sd["model_state_dict"] if "model_state_dict" in cnn_sd else cnn_sd
    )
    cnn_model.eval()
    print("✅ CNN model loaded")

    # ViT model
    vit_path = get_model_path("vit_transformer", "model.pth")
    vit_model = ViTTest(num_classes=2)
    vit_sd = torch.load(vit_path, map_location=device, weights_only=True)

    if "model_state_dict" in vit_sd:
        vit_sd = vit_sd["model_state_dict"]

    # Prilagodba ključeva
    if any(
        k.startswith("backbone.") for k in vit_model.state_dict().keys()
    ) and not any(k.startswith("backbone.") for k in vit_sd.keys()):
        vit_sd = {"backbone." + k: v for k, v in vit_sd.items()}

    vit_model.load_state_dict(vit_sd)
    vit_model.eval()
    print("✅ ViT model loaded")

    # Inicijalizacija generatora vizualizacija
    inner_model = vit_model.backbone if hasattr(vit_model, "backbone") else vit_model
    gradcam_gen = GradCamGenerator(cnn_model)

    try:
        vit_attention_gen = ViTAttentionGenerator(inner_model)
        print("✅ ViT Attention Generator ready (timm-specific)")
    except Exception as e:
        print(f"⚠️ Could not initialize ViT Attention: {e}")
        vit_attention_gen = None

    print("✅ Visualization generators ready")
    print(
        f"ViT model blocks: {len(inner_model.blocks) if hasattr(inner_model, 'blocks') else 'N/A'}"
    )

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
async def process_cnn_prediction(input_tensor: torch.Tensor) -> dict:
    """Async obrada CNN predikcije."""
    loop = asyncio.get_event_loop()
    with torch.no_grad():
        cnn_out = await loop.run_in_executor(executor, lambda: cnn_model(input_tensor))
        cnn_probs = torch.nn.functional.softmax(cnn_out, dim=1)[0]
        cnn_conf, cnn_class = torch.max(cnn_probs, dim=0)

        return {
            "output": cnn_out,
            "probs": cnn_probs,
            "confidence": cnn_conf.item(),
            "class_idx": cnn_class.item(),
            "probabilities": cnn_probs.tolist(),
        }


async def process_vit_prediction(input_tensor: torch.Tensor) -> dict:
    """Async obrada ViT predikcije."""
    loop = asyncio.get_event_loop()
    with torch.no_grad():
        try:
            vit_out = await loop.run_in_executor(
                executor, lambda: vit_model(input_tensor)
            )
            vit_probs = torch.nn.functional.softmax(vit_out, dim=1)[0]
            vit_conf, vit_class = torch.max(vit_probs, dim=0)

            return {
                "output": vit_out,
                "probs": vit_probs,
                "confidence": vit_conf.item(),
                "class_idx": vit_class.item(),
                "probabilities": vit_probs.tolist(),
                "error": None,
            }
        except Exception as e:
            print(f"ViT prediction error: {e}")
            return {
                "output": None,
                "probs": torch.tensor([0.5, 0.5]),
                "confidence": 0.5,
                "class_idx": 0,
                "probabilities": [0.5, 0.5],
                "error": str(e),
            }


async def process_gradcam(pil_image: Image.Image, target_class: int) -> str:
    """Async generiranje Grad-CAM vizualizacije."""
    loop = asyncio.get_event_loop()
    try:
        cnn_cam = await loop.run_in_executor(
            executor, lambda: gradcam_gen.generate_single(pil_image, target_class)
        )
        return cnn_cam
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


async def process_vit_attention(pil_image: Image.Image) -> str:
    """Async generiranje ViT attention mape."""
    loop = asyncio.get_event_loop()
    try:
        vit_map = await loop.run_in_executor(
            executor, lambda: vit_attention_gen.generate_attention_map(pil_image)
        )
        return vit_map
    except Exception as e:
        print(f"ViT attention error: {e}")
        return None


@app.get("/progress/{request_id}")
async def get_progress(request_id: str):
    """Endpoint za praćenje progresa."""
    if request_id not in processing_status:
        raise HTTPException(404, "Request ID not found")
    return processing_status[request_id]


async def cleanup_status(request_id: str):
    """Čisti status nakon 5 minuta."""
    await asyncio.sleep(300)  # 5 minuta
    if request_id in processing_status:
        del processing_status[request_id]


@app.post("/predict-dual")
async def predict_dual(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = None
):
    """Glavni endpoint za predviđanje s oba modela."""
    request_id = f"{time.time()}_{hashlib.md5(file.filename.encode()).hexdigest()[:8]}"
    processing_status[request_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting image processing...",
        "filename": file.filename,
        "start_time": time.time(),
    }

    try:
        # Faza 1: Validacija i čitanje slike
        processing_status[request_id].update(
            {"progress": 10, "message": "Validating image..."}
        )

        file_ext = validate_image(file)
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                413, f"File too large. Max {MAX_FILE_SIZE//1024//1024}MB"
            )

        # Provjeri cache
        cache_key = get_cache_key(contents, "full_prediction")
        if cache_key in prediction_cache:
            processing_status[request_id].update(
                {"status": "complete", "progress": 100, "message": "Served from cache"}
            )
            return prediction_cache[cache_key]

        processing_status[request_id].update(
            {"progress": 20, "message": "Processing image..."}
        )

        # Otvaranje i validacija slike
        try:
            pil_image = Image.open(io.BytesIO(contents))
            pil_image.verify()
            pil_image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(400, f"Invalid image file: {str(e)}")

        pil_image = pil_image.convert("RGB")
        original_size = pil_image.size

        # Faza 2: Preprocesiranje
        processing_status[request_id].update(
            {"progress": 30, "message": "Preprocessing image..."}
        )

        input_tensor = transform(pil_image).unsqueeze(0)

        # Faza 3: Paralelne predikcije
        processing_status[request_id].update(
            {"progress": 40, "message": "Running CNN prediction..."}
        )

        cnn_task = asyncio.create_task(process_cnn_prediction(input_tensor))

        processing_status[request_id].update(
            {"progress": 50, "message": "Running ViT prediction..."}
        )

        vit_task = asyncio.create_task(process_vit_prediction(input_tensor))

        # Čekaj obje predikcije
        cnn_result, vit_result = await asyncio.gather(cnn_task, vit_task)

        labels = ["AI", "Real"]

        # Faza 4: Generiranje vizualizacija
        processing_status[request_id].update(
            {"progress": 70, "message": "Generating visualizations..."}
        )

        gradcam_task = asyncio.create_task(
            process_gradcam(pil_image, cnn_result["class_idx"])
        )
        attention_task = asyncio.create_task(process_vit_attention(pil_image))

        cnn_cam, vit_map = await asyncio.gather(gradcam_task, attention_task)

        # Faza 5: Priprema rezultata
        processing_status[request_id].update(
            {"progress": 90, "message": "Preparing results..."}
        )

        result = {
            "filename": file.filename,
            "file_type": file_ext,
            "image_size": f"{original_size[0]}x{original_size[1]}",
            "processed_size": "224x224",
            "request_id": request_id,
            "processing_time": round(
                time.time() - processing_status[request_id]["start_time"], 2
            ),
            # --- ISPRAVAK U main.py ---
            "cnn": {
                "label": labels[cnn_result["class_idx"]],
                "probability": round(cnn_result["confidence"], 4),
                "confidence_percent": round(cnn_result["confidence"] * 100, 1),
                "raw_probabilities": {
                    # Budući da je 0=AI a 1=Real:
                    "ai": round(cnn_result["probabilities"][0], 4),
                    "real": round(cnn_result["probabilities"][1], 4),
                },
                "visualization": cnn_cam,
            },
            "vit": {
                "label": (
                    labels[vit_result["class_idx"]]
                    if vit_result["error"] is None
                    else "Unknown"
                ),
                "probability": (
                    round(vit_result["confidence"], 4)
                    if vit_result["error"] is None
                    else 0.0
                ),
                "confidence_percent": (
                    round(vit_result["confidence"] * 100, 1)
                    if vit_result["error"] is None
                    else 0.0
                ),
                "raw_probabilities": {
                    # Budući da je 0=AI a 1=Real:
                    "ai": (
                        round(vit_result["probabilities"][0], 4)
                        if vit_result["error"] is None
                        else 0.5
                    ),
                    "real": (
                        round(vit_result["probabilities"][1], 4)
                        if vit_result["error"] is None
                        else 0.5
                    ),
                },
                "visualization": vit_map,
                "error": vit_result["error"],
            },
        }

        # Dodaj komparaciju
        if vit_result["error"] is None:
            models_agree = (
                labels[cnn_result["class_idx"]] == labels[vit_result["class_idx"]]
            )
            result["comparison"] = {
                "models_agree": models_agree,
                "agreement": "Both models agree" if models_agree else "Models disagree",
                "confidence_difference": round(
                    abs(cnn_result["confidence"] - vit_result["confidence"]) * 100, 1
                ),
            }
        else:
            result["comparison"] = {
                "models_agree": False,
                "agreement": "ViT model failed",
                "confidence_difference": None,
            }

        # Spremi u cache
        prediction_cache[cache_key] = result

        # Limitiraj cache veličinu
        if len(prediction_cache) > 100:
            oldest_key = next(iter(prediction_cache))
            del prediction_cache[oldest_key]

        # Finalni status
        processing_status[request_id].update(
            {
                "status": "complete",
                "progress": 100,
                "message": "Analysis complete",
                "processing_time": result["processing_time"],
            }
        )

        # Zakazivanje čišćenja
        if background_tasks:
            background_tasks.add_task(cleanup_status, request_id)

        print(
            f"✅ Analysis complete for {file.filename} in {result['processing_time']}s"
        )
        return result

    except HTTPException:
        processing_status[request_id].update(
            {"status": "error", "message": "Validation error"}
        )
        raise

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Prediction error: {error_msg}")

        processing_status[request_id].update({"status": "error", "message": error_msg})

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": error_msg,
                "request_id": request_id,
            },
        )


@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """Batch endpoint za više slika (max 3)."""
    if len(files) > 3:
        raise HTTPException(400, "Maximum 3 images allowed per batch")

    results = []
    for file in files:
        try:
            # Koristimo isti endpoint ali sa BackgroundTasks
            result = await predict_dual(file)
            results.append(result)
        except Exception as e:
            results.append(
                {"filename": file.filename, "error": str(e), "status": "failed"}
            )

    return {
        "batch_id": f"batch_{int(time.time())}",
        "total_images": len(files),
        "successful": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r]),
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn

    # Konfiguracija za bolje performanse
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,  # Koristi 1 worker zbog GPU/CPU memory sharing
        "loop": "asyncio",
        "http": "httptools",
        "ws": "websockets",
        "lifespan": "on",
        "access_log": True,
        "timeout_keep_alive": 30,
    }

    print(f"🚀 Starting server on {config['host']}:{config['port']}")
    print(f"📡 CORS enabled for: {FRONTEND_URL}")
    print(f"💾 Device: {device}")
    print(f"🧠 Models: CNN={cnn_model is not None}, ViT={vit_model is not None}")

    uvicorn.run(
        "main:app",
        host=config["host"],
        port=config["port"],
        reload=True,
        reload_dirs=["."],
    )
