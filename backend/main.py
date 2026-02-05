import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from services.model_loader import initialize_all_models
from services.inference_service import InferenceService
from services.xai_service import XAIService

# --- INICIJALIZACIJA APLIKACIJE ---
app = FastAPI(title="AI Image Detector Backend", version="3.0.0")
load_dotenv()

# --- KONFIGURACIJA ---
IS_HF_SPACES = os.environ.get("SPACE_ID") is not None or "hf.space" in os.environ.get(
    "HOSTNAME", ""
)

REPO_ID = os.getenv("REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

if not REPO_ID:
    raise RuntimeError("❌ REPO_ID environment variable is not set.")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# OPTIMIZACIJA: Bolje koristi CPU za AMD GPU (torch ne podržava AMD ROCm dobro)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    # Optimiziraj CUDA settings
    torch.backends.cudnn.benchmark = True  # Brže za CNN modele
    torch.backends.cudnn.deterministic = False
else:
    print("⚠️ Using CPU - Consider using ONNX for better performance")
    # Za CPU, koristi MKL optimizacije
    torch.set_num_threads(os.cpu_count() or 4)

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

# --- GLOBALNE INSTANCE ---
global_models = None
transform = None
executor = None
inference_service = None
xai_service = None


# --- INICIJALIZACIJA ---
@app.on_event("startup")
async def startup_event():
    global global_models, transform, executor, inference_service, xai_service

    print(f"🔄 Initializing AI models on device: {device}...")
    print(f"📊 Environment: {'HF Spaces' if IS_HF_SPACES else 'Local Development'}")

    try:
        # Učitaj modele
        global_models = initialize_all_models(device, IS_HF_SPACES, REPO_ID, HF_TOKEN)

        # OPTIMIZACIJA: Povećaj broj radnika za paralelno procesiranje
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))

        # Transformacija
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Inicijaliziraj servise
        inference_service = InferenceService(global_models)
        xai_service = XAIService(global_models)

        print(f"\n✨ All services initialized successfully")
        print(f"   Device: {device}")
        print(
            f"   Models loaded: {len([m for m in global_models.values() if m is not None])}"
        )
        print(f"   Thread workers: {executor._max_workers}")
        print(f"   InferenceService: {'✓' if inference_service else '✗'}")
        print(f"   XAIService: {'✓' if xai_service else '✗'}")

    except Exception as e:
        print(f"❌ Startup initialization error: {e}")
        import traceback

        traceback.print_exc()
        raise


# --- INCLUDE ROUTES ---
from api.routes_debug import router as debug_router
from api.routes_predict import router as predict_router
from api.routes_models import router as models_router

app.include_router(debug_router)
app.include_router(predict_router)
app.include_router(models_router)


# --- DEPENDENCY INJECTION ---
@app.middleware("http")
async def add_globals_to_request(request, call_next):
    request.state.models = global_models
    request.state.transform = transform
    request.state.executor = executor
    request.state.device = device
    request.state.inference_service = inference_service
    request.state.xai_service = xai_service
    response = await call_next(request)
    return response


# --- ROOT ENDPOINT ---
@app.get("/")
async def root():
    models_status = {}
    if global_models:
        models_status = {
            "cnn_fine_tuned": global_models.get("cnn_fine_tuned") is not None,
            "vit_fine_tuned": global_models.get("vit_fine_tuned") is not None,
            "cnn_raw": global_models.get("cnn_raw") is not None,
            "vit_raw": global_models.get("vit_raw") is not None,
            "deit_fine_tuned": global_models.get("deit_fine_tuned") is not None,
            "convnext_fine_tuned": global_models.get("convnext_fine_tuned") is not None,
        }

    return {
        "status": "online",
        "service": "AI Image Detector API",
        "version": "3.0.0",
        "environment": "HF Spaces" if IS_HF_SPACES else "Local",
        "architecture": "Modular (services + API routes)",
        "device": str(device),
        "models_loaded": models_status,
        "services": {
            "InferenceService": inference_service is not None,
            "XAIService": xai_service is not None,
        },
    }


# --- MAIN ---
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
