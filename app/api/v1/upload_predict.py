from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import io
import numpy as np
from PIL import Image
import base64
from torchvision import transforms
from app.models.efficientnet_test import EfficientNetTest
from app.gradcam.gradcam_utils import GradCamGenerator  # tvoj postojeći


router = APIRouter()

# Globalni model + GradCAM
model = EfficientNetTest(num_classes=2)
model.load_trained_weights("models/efficientnet_test_mini.pth")
gradcam_gen = GradCamGenerator(model)  # već radi!


@router.post("/batch-predict-cam")
async def batch_predict_cam(images: list[UploadFile] = File(...)):
    """Batch predikcija + Grad-CAM base64 za React."""
    if not images:
        raise HTTPException(400, "No images")

    results = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for image_file in images:
        # Process image
        contents = await image_file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Predikcija
        input_tensor = transform(pil_image).unsqueeze(0)
        ai_prob = model.predict_test(input_tensor)
        
        # 🎯 GRAD-CAM
        cam_b64 = gradcam_gen.generate_single(pil_image, target_class=0)
        
        results.append({
            "filename": image_file.filename,
            "label": "AI" if ai_prob > 0.5 else "Real",
            "ai_probability": round(ai_prob, 4),
            "gradcam_b64": cam_b64  # za React <img src={`data:image/png;base64,${b64}`} />
        })

    return {"results": results}
