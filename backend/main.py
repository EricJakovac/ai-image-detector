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
# AUTO-DETECT: JESMO LI NA HF SPACES ILI LOKALNO?
IS_HF_SPACES = os.environ.get("SPACE_ID") is not None or "hf.space" in os.environ.get(
    "HOSTNAME", ""
)

# MODEL REPO - PRODUKCIJA KORISTI HF HUB, LOKALNO KORISTI LOKALNE
if IS_HF_SPACES:
    # Na HF Spaces: koristi modele iz HF Hub-a
    REPO_ID = "EricJakovac/ai-image-detector-models"  # Tvoj model repo na HF Hubu
    print("🚀 RUNNING ON HF SPACES - Using HF Hub models")
else:
    # Lokalno: koristi environment varijable ili fallback
    REPO_ID = os.getenv("REPO_ID", "EricJakovac/ai-image-detector-models")
    print("💻 RUNNING LOCALLY - Checking for local models first")

HF_TOKEN = os.getenv("HF_TOKEN")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Automatsko prebacivanje na GPU ako je dostupan, inače CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
executor = ThreadPoolExecutor(max_workers=2)
processing_status: Dict[str, dict] = {}
prediction_cache: Dict[str, dict] = {}

# --- CORS KONFIGURACIJA ---
# Podesi različito za produkciju i development
if IS_HF_SPACES:
    # Na HF Spaces: dopusti sve ili specifične domene
    allow_origins = ["*"]  # Ili: ["https://*.hf.space", FRONTEND_URL]
else:
    # Lokalno: samo frontend i localhost
    allow_origins = [FRONTEND_URL, "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- POMOĆNE FUNKCIJE (Ažurirane) ---
def get_model_path_optimized(folder_name: str, filename: str) -> str:
    # Definiramo putanju unutar HF cache-a koja preživljava većinu restarta
    # Na HF Spaces, /home/user/.cache je obično postojaniji od samog /app foldera
    hf_cache_path = (
        "/home/user/.cache/huggingface/hub" if IS_HF_SPACES else "./model_cache"
    )

    local_path = os.path.join("models", folder_name, filename)

    # 1. Provjera: Ako datoteka već postoji u našem lokalnom folderu i ispravna je (>1MB)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000000:
        print(f"✔️ Using existing model from local folder: {local_path}")
        return local_path

    # 2. Provjera: hf_hub_download po defaultu već ima ugrađen sustav koji NE skida
    # datoteku ako je već u cache-u i ista je kao na Hub-u.
    print(f"🔍 Checking/Fetching model: {folder_name}/{filename}")

    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"{folder_name}/{filename}",
            token=HF_TOKEN,
            cache_dir=hf_cache_path,
            local_files_only=False,  # On će sam provjeriti ETag (hash) i skinuti samo ako ima promjena
        )
        print(f"✅ Model path ready: {path}")
        return path
    except Exception as e:
        # Ako nema interneta, a imamo nešto u cache-u, pokušaj bar to
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


def get_cache_key(file_contents: bytes, model_name: str) -> str:
    """Generiraj cache key od fajla."""
    file_hash = hashlib.md5(file_contents).hexdigest()
    return f"{model_name}_{file_hash}"


# --- INICIJALIZACIJA MODELA (Optimized) ---
print(f"🔄 Initializing AI models on device: {device}...")
print(f"📊 Environment: {'HF Spaces' if IS_HF_SPACES else 'Local Development'}")

# CNN (EfficientNet) model
cnn_model = None
vit_model = None
gradcam_gen = None
vit_attention_gen = None

try:
    # CNN model - koristi optimizirani loader
    cnn_path = get_model_path_optimized("cnn_efficientnet", "model.pth")
    cnn_model = EfficientNetTest(num_classes=2)
    cnn_sd = torch.load(cnn_path, map_location=device, weights_only=False)
    cnn_model.load_state_dict(
        cnn_sd["model_state_dict"] if "model_state_dict" in cnn_sd else cnn_sd
    )
    cnn_model.to(device)
    cnn_model.eval()
    print("✅ CNN model loaded")

    # ViT model - koristi optimizirani loader
    vit_path = get_model_path_optimized("vit_transformer", "model.pth")
    vit_model = ViTTest(num_classes=2)
    vit_sd = torch.load(vit_path, map_location=device, weights_only=False)

    if "model_state_dict" in vit_sd:
        vit_sd = vit_sd["model_state_dict"]

    # Prilagodba ključeva
    if any(
        k.startswith("backbone.") for k in vit_model.state_dict().keys()
    ) and not any(k.startswith("backbone.") for k in vit_sd.keys()):
        vit_sd = {"backbone." + k: v for k, v in vit_sd.items()}

    vit_model.load_state_dict(vit_sd)
    vit_model.to(device)
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

    print("✅ All models initialized successfully")

except Exception as e:
    print(f"❌ Model initialization error: {e}")
    import traceback

    traceback.print_exc()
    raise

# --- Ostali dijelovi koda ostaju ISTI ---
# (transform, validate_image, async funkcije, endpointi...)

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
        cnn_out = await loop.run_in_executor(
            executor, lambda: cnn_model(input_tensor.to(device))
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


async def process_vit_prediction(input_tensor: torch.Tensor) -> dict:
    """Async obrada ViT predikcije."""
    loop = asyncio.get_event_loop()
    with torch.no_grad():
        try:
            vit_out = await loop.run_in_executor(
                executor, lambda: vit_model(input_tensor.to(device))
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


# --- ENDPOINTI (dodaj debug endpoint) ---
@app.get("/debug/environment")
async def debug_environment():
    """Debug endpoint za provjeru okoline i modela."""
    local_cnn = os.path.join("models", "cnn_efficientnet", "model.pth")
    local_vit = os.path.join("models", "vit_transformer", "model.pth")

    return {
        "environment": {
            "is_hf_spaces": IS_HF_SPACES,
            "space_id": os.environ.get("SPACE_ID"),
            "hostname": os.environ.get("HOSTNAME"),
            "repo_id": REPO_ID,
            "device": str(device),
        },
        "local_files": {
            "cnn_exists": os.path.exists(local_cnn),
            "cnn_size_mb": (
                os.path.getsize(local_cnn) / (1024 * 1024)
                if os.path.exists(local_cnn)
                else 0
            ),
            "vit_exists": os.path.exists(local_vit),
            "vit_size_mb": (
                os.path.getsize(local_vit) / (1024 * 1024)
                if os.path.exists(local_vit)
                else 0
            ),
        },
        "models_loaded": {
            "cnn": cnn_model is not None,
            "vit": vit_model is not None,
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

        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
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
            "cnn": {
                "label": labels[cnn_result["class_idx"]],
                "probability": round(cnn_result["confidence"], 4),
                "confidence_percent": round(cnn_result["confidence"] * 100, 1),
                "raw_probabilities": {
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

        # Spremi u cache
        prediction_cache[cache_key] = result

        # Limitiraj cache veličinu
        if len(prediction_cache) > 100:
            oldest_key = next(iter(prediction_cache))
            del prediction_cache[oldest_key]

        # Finalni status
        processing_status[request_id].update(
            {"status": "complete", "progress": 100, "message": "Analysis complete"}
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
            result = await predict_dual(file)
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
        "version": "1.0.0",
        "environment": "HF Spaces" if IS_HF_SPACES else "Local",
        "endpoints": {
            "predict": "/predict-dual (POST)",
            "progress": "/progress/{request_id} (GET)",
            "batch": "/batch-predict (POST)",
            "debug": "/debug/environment (GET)",
        },
        "models_loaded": {
            "cnn": cnn_model is not None,
            "vit": vit_model is not None,
            "gradcam": gradcam_gen is not None,
            "vit_attention": vit_attention_gen is not None,
        },
        "device": str(device),
    }


if __name__ == "__main__":
    import uvicorn

    # Dinamički dohvaćamo port: na Hugging Face će biti sistemski PORT (npr. 7860), lokalno 8000
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True if port == 8000 else False,  # Reload samo lokalno
        workers=1,
    )
