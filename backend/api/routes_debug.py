from fastapi import APIRouter, HTTPException, Request
import os

router = APIRouter()


@router.get("/environment")
async def debug_environment(request: Request):
    """Debug endpoint za provjeru okoline i modela."""
    # Dobij podatke iz request state
    models = request.state.models
    xai_generators = getattr(request.state, "xai_generators", {})

    from main import IS_HF_SPACES, REPO_ID, device

    return {
        "environment": {
            "is_hf_spaces": IS_HF_SPACES,
            "space_id": os.environ.get("SPACE_ID"),
            "hostname": os.environ.get("HOSTNAME"),
            "repo_id": REPO_ID,
            "device": str(device),
        },
        "models_loaded": {
            "cnn_fine_tuned": (
                models.get("cnn_fine_tuned") is not None if models else False
            ),
            "vit_fine_tuned": (
                models.get("vit_fine_tuned") is not None if models else False
            ),
            "cnn_raw": models.get("cnn_raw") is not None if models else False,
            "vit_raw": models.get("vit_raw") is not None if models else False,
            "deit_fine_tuned": (
                models.get("deit_fine_tuned") is not None if models else False
            ),
            "convnext_fine_tuned": (
                models.get("convnext_fine_tuned") is not None if models else False
            ),
        },
        "services": {
            "inference_service": request.state.inference_service is not None,
            "xai_service": request.state.xai_service is not None,
        },
        "xai_generators": {
            "cnn_fine_tuned": "cnn_fine_tuned" in xai_generators,
            "cnn_raw": "cnn_raw" in xai_generators,
            "vit_fine_tuned": "vit_fine_tuned" in xai_generators,
            "vit_raw": "vit_raw" in xai_generators,
            "deit_fine_tuned": "deit_fine_tuned" in xai_generators,
            "convnext_fine_tuned": "convnext_fine_tuned" in xai_generators,
        },
    }


@router.get("/progress/{request_id}")
async def get_progress(request_id: str):
    """Endpoint za praćenje progresa."""
    from utils.cache_utils import processing_status

    if request_id not in processing_status:
        raise HTTPException(404, "Request ID not found")
    return processing_status[request_id]
