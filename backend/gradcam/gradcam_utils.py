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

        # Target layer za EfficientNet
        target_module = model.backbone.conv_head
        self.target_layers = [target_module]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        print(f"✅ Grad-CAM generator inicijaliziran za EfficientNet")

    def generate_single(self, pil_image, target_class=0):
        """Grad-CAM → base64 PNG."""
        # Resize za model
        resized_img = pil_image.resize((224, 224))

        # Input tensor
        input_tensor = F.to_tensor(resized_img).unsqueeze(0)
        input_tensor = F.normalize(
            input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

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
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class GradCamGeneratorRaw:
    def __init__(self, model):
        self.model = model.eval()

        # Pronađi zadnji Conv2D layer
        conv_layers = []

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((name, module))

        if conv_layers:
            # Koristi zadnji conv layer
            last_conv_name, last_conv_module = conv_layers[-1]
            target_module = last_conv_module
            print(f"✅ Target layer: {last_conv_name}")
        else:
            # Fallback: pronađi bilo koji layer
            all_modules = list(model.named_modules())
            if all_modules:
                last_module_name, last_module = all_modules[-1]
                target_module = last_module
                print(f"⚠️ Nema Conv2D layera, koristim: {last_module_name}")
            else:
                print("❌ Nema modula u modelu!")
                target_module = None

        if target_module is None:
            print("❌ Nije pronađen target layer!")
            self.cam = None
            return

        self.target_layers = [target_module]

        try:
            # VAŽNO: Omogući gradijente za target layer
            for param in self.model.parameters():
                param.requires_grad = True

            self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        except Exception as e:
            print(f"❌ Greška pri inicijalizaciji Grad-CAM: {e}")
            self.cam = None

    def generate_single(self, pil_image, target_class=0):
        """Grad-CAM za RAW model → base64 PNG."""
        if self.cam is None:
            print("❌ Grad-CAM nije inicijaliziran!")
            return None

        if self.model is None:
            print("❌ Model nije postavljen!")
            return None

        try:
            # Pripremi sliku
            resized_img = pil_image.resize((224, 224))

            # Kreiraj input tensor
            input_tensor = F.to_tensor(resized_img).unsqueeze(0)
            input_tensor = F.normalize(
                input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            # VAŽNO: Omogući gradijente za forward pass
            input_tensor.requires_grad = True

            # Testiraj model
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(input_tensor)

                if target_class >= outputs.shape[1]:
                    print(f"⚠️ Target class {target_class} prevelik, koristim class 0")
                    target_class = 0

            # Generiraj CAM
            targets = [ClassifierOutputTarget(target_class)]

            # VAŽNO: Omogući gradijente za CAM generaciju
            with torch.enable_grad():
                grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)

            if grayscale_cam is None:
                print("❌ CAM generacija nije uspjela!")
                return None

            grayscale_cam = grayscale_cam[0, :]

            # Overlay
            rgb_img = np.array(resized_img).astype(np.float32) / 255.0
            cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            if cam_overlay is None:
                print("❌ Overlay nije uspio!")
                return None

            # Base64
            cam_pil = Image.fromarray((cam_overlay * 255).astype(np.uint8))
            buffered = BytesIO()
            cam_pil.save(buffered, format="PNG")
            result = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return result

        except Exception as e:
            print(f"❌ Greška pri generiranju Grad-CAM: {e}")
            return None
