import torch
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from torchvision import transforms


class DeiTAttentionGenerator:
    """Attention generator za DeiT Transformer - identičan ViT-u"""

    def __init__(self, model):
        self.model = model.eval()
        self.attention_maps = []
        self.hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        """Postavi hooks na DeiT attention layere"""
        self.hooks = []
        # DeiT ima sličnu strukturu kao ViT
        for name, module in self.model.named_modules():
            if "blocks" in name and name.endswith("norm1"):
                self.hooks.append(module.register_forward_hook(self._hook_fn))

        print(f"✅ DeiT Hooks registrirani na {len(self.hooks)} layera")

    def _hook_fn(self, module, input, output):
        """Hook funkcija za hvatanje outputa"""
        self.attention_maps.append(output.detach())

    def generate_attention_map(self, pil_image):
        """Generiraj attention mapu za DeiT"""
        self.attention_maps = []

        try:
            # Pripremi sliku
            img_224 = pil_image.resize((224, 224))

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            input_tensor = transform(img_224).unsqueeze(0)

            # Forward pass
            with torch.no_grad():
                _ = self.model(input_tensor)

            if not self.attention_maps:
                print("⚠️ Nema attention mapa za DeiT")
                return None

            # Uzmi zadnji sloj
            feat = self.attention_maps[-1]  # Shape: [1, 197, 768]

            # Izračunaj važnost svakog patcha (L2 norma)
            patch_importance = torch.norm(feat[0, 1:, :], dim=1)  # [196]

            # Reshape u grid 14x14
            grid_size = 14
            importance_map = (
                patch_importance.reshape(grid_size, grid_size).cpu().numpy()
            )

            # Normalizacija
            importance_map = (importance_map - importance_map.min()) / (
                importance_map.max() - importance_map.min() + 1e-8
            )

            # Resize na originalnu veličinu
            importance_map = cv2.resize(importance_map, (224, 224))

            # Kreiraj toplinsku mapu
            heatmap = cv2.applyColorMap(
                np.uint8(255 * importance_map), cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Preklapanje s originalnom slikom
            img_np = np.array(img_224)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

            # Konverzija u Base64
            res_pil = Image.fromarray(overlay)
            buffered = BytesIO()
            res_pil.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        except Exception as e:
            print(f"❌ Greška pri generiranju DeiT attention: {e}")
            return None

    def cleanup(self):
        """Očisti hooks"""
        for hook in self.hooks:
            hook.remove()
