import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import io
import base64


class GradCamHandler:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # NAĐI TOČAN TARGET LAYER (zadnji conv)
        target_layer_name = self._find_target_layer()
        print(f"✅ Koristi layer: {target_layer_name}")
        
        # Dohvati PRAVI model.layer objekt (ne string!)
        target_layer = dict(model.named_modules())[target_layer_name]
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])

    def _find_target_layer(self):
        """Automatski pronađi zadnji Conv2d layer."""
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = name
        return last_conv

    def generate_gradcam(self, image_pil, target_class=0):
        """Generira Grad-CAM za sliku. Vraća base64 PNG."""
        # Preprocess
        input_tensor = F.to_tensor(image_pil).unsqueeze(0)
        
        # Grad-CAM
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Overlay
        rgb_img = np.array(image_pil) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Base64 za frontend
        buffered = io.BytesIO()
        Image.fromarray(cam_image).save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
