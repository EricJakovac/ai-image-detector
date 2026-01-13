import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import base64
from io import BytesIO
from PIL import Image
from torchvision import transforms




class GradCamGenerator:
    def __init__(self, model):
        self.model = model.eval()
        # 🎯 PRAVI FIX: modul objekt, ne string!
        target_layer = model.backbone.conv_head  # ili model.backbone.features[-1]
        self.target_layers = [target_layer]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        print(f"✅ Grad-CAM target: {target_layer}")
    
    def generate_single(self, pil_image, target_class=0):  # 0 = AI
        """Generira Grad-CAM za JEDNU sliku."""
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(pil_image).unsqueeze(0)
        
        # Grad-CAM
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets
        )[0, :]
        
        # Overlay na original
        rgb_img = np.array(pil_image.resize((224, 224))) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Base64 za React
        pil_vis = Image.fromarray((visualization * 255).astype(np.uint8))
        buffered = BytesIO()
        pil_vis.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
