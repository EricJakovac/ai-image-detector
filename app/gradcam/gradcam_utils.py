import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import base64
from io import BytesIO
from PIL import Image
import torchvision.transforms.functional as F


class GradCamGenerator:
    def __init__(self, model):
        self.model = model.eval()
        
        # 🎯 TOČAN FIX: modul objekt (ne string!)
        target_module = model.backbone.conv_head
        self.target_layers = [target_module]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        print(f"✅ Grad-CAM spreman za: {target_module}")

    def generate_single(self, pil_image, target_class=0):
        """Grad-CAM → base64 PNG."""
        # Resize za model
        resized_img = pil_image.resize((224, 224))
        
        # Input tensor
        input_tensor = F.to_tensor(resized_img).unsqueeze(0)
        input_tensor = F.normalize(input_tensor, 
                                  mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
        
        # Generiraj CAM
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Overlay
        rgb_img = np.array(resized_img) / 255.0
        cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Base64
        cam_pil = Image.fromarray((cam_overlay * 255).astype(np.uint8))
        buffered = BytesIO()
        cam_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
