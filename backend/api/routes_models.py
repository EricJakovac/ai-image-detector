from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/info")
async def models_info(request: Request):
    """Informacije o trenutno učitanim modelima."""
    # Dobij modele iz request state
    models = request.state.models

    return {
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
        "available_models": [
            {
                "name": "cnn_fine_tuned",
                "type": "fine_tuned",
                "architecture": "EfficientNet",
            },
            {"name": "cnn_raw", "type": "raw", "architecture": "EfficientNet"},
            {
                "name": "vit_fine_tuned",
                "type": "fine_tuned",
                "architecture": "Vision Transformer",
            },
            {"name": "vit_raw", "type": "raw", "architecture": "Vision Transformer"},
            {"name": "deit_fine_tuned", "type": "fine_tuned", "architecture": "DeiT"},
            {
                "name": "convnext_fine_tuned",
                "type": "fine_tuned",
                "architecture": "ConvNeXt",
            },
        ],
    }
