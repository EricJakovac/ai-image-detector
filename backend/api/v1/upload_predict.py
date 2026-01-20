from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import base64
from torchvision import transforms
from models.efficientnet_test import EfficientNetTest
from gradcam.gradcam_utils import GradCamGenerator

router = APIRouter()

# Globalni model i GradCAM
model = EfficientNetTest(num_classes=2)
model.load_trained_weights("models/efficientnet_test_mini.pth")
gradcam_gen = GradCamGenerator(model)

# Transformacija za inferenciju
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@router.post("/batch-predict-cam")
async def batch_predict_cam(images: list[UploadFile] = File(...)):
    """Batch predikcija + Grad-CAM za svaku sliku."""
    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded")

    results = []

    for image_file in images:
        # Process slika
        contents = await image_file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 1. Predikcija
        input_tensor = transform(pil_image).unsqueeze(0)
        ai_prob = model.predict_test(input_tensor)

        # 2. Grad-CAM
        try:
            cam_b64 = gradcam_gen.generate_single(pil_image, target_class=0)
        except Exception as e:
            cam_b64 = f"Grad-CAM error: {str(e)}"

        results.append(
            {
                "filename": image_file.filename,
                "label": "AI" if ai_prob > 0.5 else "Real",
                "ai_probability": round(float(ai_prob), 4),
                "gradcam_b64": cam_b64,  # PNG base64 za <img src="data:image/png;base64,{cam_b64}"/>
            }
        )

    return {"results": results, "batch_size": len(images)}


@router.get("/test-cam")
def test_cam():
    return {
        "message": "Grad-CAM endpoint spreman, pošalji form-data sa 'images' field-om"
    }
