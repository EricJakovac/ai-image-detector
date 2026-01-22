import torch
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from torchvision import transforms


class ViTAttentionGenerator:
    def __init__(self, model):
        self.model = model.eval()
        self.attention_maps = []
        self.hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        self.hooks = []
        # Registriramo hook na zadnji norm sloj bloka jer on sadrži finalne značajke prije klasifikacije
        for name, module in self.model.named_modules():
            if "blocks" in name and name.endswith("norm1"):
                self.hooks.append(module.register_forward_hook(self._hook_fn))
        print(f"✅ ViT Hooks registrirani na {len(self.hooks)} slojeva.")

    def _hook_fn(self, module, input, output):
        self.attention_maps.append(output.detach())

    def generate_attention_map(self, pil_image):
        self.attention_maps = []
        img_224 = pil_image.resize((224, 224))

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = transform(img_224).unsqueeze(0)

        with torch.no_grad():
            _ = self.model(input_tensor)

        if not self.attention_maps:
            return None

        # Uzmi zadnji sloj (najdublje razumijevanje slike)
        # Shape: [1, 197, 768]
        feat = self.attention_maps[-1]

        # 1. Izračunaj važnost svakog patcha (L2 norma preko 768 dimenzija)
        # Ignoriramo prvi token (CLS token) na indeksu 0
        patch_importance = torch.norm(feat[0, 1:, :], dim=1)  # Rezultat: [196]

        # 2. Reshape u grid 14x14
        grid_size = 14
        importance_map = patch_importance.reshape(grid_size, grid_size).cpu().numpy()

        # 3. Normalizacija i skaliranje
        importance_map = (importance_map - importance_map.min()) / (
            importance_map.max() - importance_map.min() + 1e-8
        )

        # 4. Resize na originalnu veličinu i kreiranje toplinske mape
        importance_map = cv2.resize(importance_map, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * importance_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 5. Preklapanje s originalnom slikom
        img_np = np.array(img_224)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        # 6. Konverzija u Base64
        res_pil = Image.fromarray(overlay)
        buffered = BytesIO()
        res_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def cleanup(self):
        for h in self.hooks:
            h.remove()
