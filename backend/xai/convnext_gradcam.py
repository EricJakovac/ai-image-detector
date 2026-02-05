import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import base64
from io import BytesIO
from PIL import Image
import torchvision.transforms.functional as F


class ConvNeXtGradCamGenerator:
    """Grad-CAM generator za ConvNeXt Tiny"""

    def __init__(self, model):
        self.model = model.eval()

        # Pronađi target layer za ConvNeXt
        # ConvNeXt ima 'stages' sa 'blocks' unutra
        target_layer = self._find_convnext_target_layer()

        if target_layer is None:
            raise ValueError("❌ Nije pronađen target layer za ConvNeXt")

        print(f"✅ ConvNeXt Grad-CAM inicijaliziran")
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])

    def _find_convnext_target_layer(self):
        """Pronađi zadnji Conv2D layer u ConvNeXt modelu"""

        # ConvNeXt struktura: stem -> stages[0-3] -> norm -> head
        # Tražimo zadnji Conv2D layer u zadnjem stage-u

        last_conv_layer = None

        for name, module in self.model.named_modules():
            # ConvNeXt specific layers
            if "stages.3" in name or "stages.2" in name:
                if isinstance(module, torch.nn.Conv2d):
                    last_conv_layer = module

            # Fallback: bilo koji Conv2D u zadnjem dijelu modela
            elif isinstance(module, torch.nn.Conv2d):
                last_conv_layer = module

        if last_conv_layer is None:
            # Ako nismo našli, uzmemo zadnji layer
            all_modules = list(self.model.named_modules())
            for name, module in reversed(all_modules):
                if isinstance(module, torch.nn.Conv2d) or isinstance(
                    module, torch.nn.Linear
                ):
                    last_conv_layer = module
                    print(f"  ⚠️ Fallback layer: {name}")
                    break

        return last_conv_layer

    def generate_single(self, pil_image, target_class=0):
        """Grad-CAM → base64 PNG za ConvNeXt."""
        try:
            # Resize za model
            resized_img = pil_image.resize((224, 224))

            # Input tensor
            input_tensor = F.to_tensor(resized_img).unsqueeze(0)
            input_tensor = F.normalize(
                input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            # Generiraj CAM
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)

            if grayscale_cam is None:
                print("❌ CAM generacija nije uspjela za ConvNeXt")
                return None

            grayscale_cam = grayscale_cam[0, :]

            # Overlay na sliku
            rgb_img = np.array(resized_img).astype(np.float32) / 255.0

            # Provjeri dimenzije
            if grayscale_cam.shape != rgb_img.shape[:2]:
                print(f"⚠️ Reshaping CAM: {grayscale_cam.shape} -> {rgb_img.shape[:2]}")
                import cv2

                grayscale_cam = cv2.resize(
                    grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0])
                )

            cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # Base64
            cam_pil = Image.fromarray((cam_overlay * 255).astype(np.uint8))
            buffered = BytesIO()
            cam_pil.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        except Exception as e:
            print(f"❌ Greška pri generiranju Grad-CAM za ConvNeXt: {e}")
            import traceback

            traceback.print_exc()
            return None
