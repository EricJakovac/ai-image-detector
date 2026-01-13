from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
import numpy as np
from torchvision import transforms
from app.models.efficientnet_test import EfficientNetTest


router = APIRouter()

# Globalni model sa treniranim težinama
model = EfficientNetTest(num_classes=2)
model.load_trained_weights("models/efficientnet_test_mini.pth")


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@router.post("/batch-predict")
async def batch_predict(images: list[UploadFile] = File(...)):
    """
    Batch predikcija: uploadaj više slika, dobij rezultate za sve.
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded")
    
    batch_tensors = []
    
    for i, image in enumerate(images):
        # Provjeri format
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"Image {i}: Invalid format")
        
        # Process sliku
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(pil_image).unsqueeze(0)  # [1, 3, 224, 224]
        batch_tensors.append(tensor)
    
    # Batch inferencija
    batch_tensor = torch.cat(batch_tensors)  # [N, 3, 224, 224]
    ai_probs = model.batch_predict(batch_tensor)
    
    results = []
    for i, prob in enumerate(ai_probs):
        results.append({
            "image_filename": images[i].filename,
            "label": "AI" if prob > 0.5 else "Real",
            "ai_probability": round(prob, 4)
        })
    
    return {"results": results, "batch_size": len(images)}
