from fastapi import APIRouter
from app.models.efficientnet_test import EfficientNetTest
import torch

router = APIRouter()

# Globalni model instance (učita se na startup)
model = EfficientNetTest(num_classes=2, pretrained=True)


@router.get("/ping")
def ping():
    return {"message": "test endpoint OK"}


@router.get("/test-predict")
def test_predict():
    """
    Test endpoint: simulira predikciju sa random slikom tensorom.
    """
    # Kreiraj dummy input tensor za test (224x224 standard)
    dummy_input = torch.randn(1, 3, 224, 224)
    ai_prob = model.predict_test(dummy_input)
    return {
        "model": "efficientnet_test",
        "label": "AI" if ai_prob > 0.5 else "real",
        "ai_probability": ai_prob,
    }
