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


class ViTAttentionGeneratorRaw:
    def __init__(self, model):
        self.model = model.eval()
        self.attention_maps = []
        self.hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        self.hooks = []
        # 🎯 KLJUČNA RAZLIKA: RAW ViT ima drugačiju strukturu!
        # Probaj različite layere
        for name, module in self.model.named_modules():
            # RAW ViT obično ima "blocks" ili "encoder.layers"
            if "blocks" in name and "attention" in name and "dropout" not in name:
                self.hooks.append(module.register_forward_hook(self._hook_fn))
            elif "encoder.layer" in name and "attention" in name:
                self.hooks.append(module.register_forward_hook(self._hook_fn))

        if not self.hooks:
            # Ako nismo našli, probajmo na norm layere
            for name, module in self.model.named_modules():
                if "norm" in name and name.endswith("1"):
                    self.hooks.append(module.register_forward_hook(self._hook_fn))

        print(f"✅ ViT RAW Hooks registrirani na {len(self.hooks)} slojeva.")

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
            # Pokušaj alternativnom metodom
            return self._generate_fallback_attention(pil_image)

        # Uzmi zadnji attention sloj
        # Shape može biti različit: [1, num_heads, seq_len, seq_len] ili [1, seq_len, dim]
        feat = self.attention_maps[-1]

        # Ako je attention sa headovima, prosječi ih
        if len(feat.shape) == 4:
            feat = feat.mean(dim=1)  # [1, seq_len, seq_len]

        # Uzmi attention prema CLS tokenu (prvi token)
        if len(feat.shape) == 3:
            # feat[0, 0, 1:] = attention od CLS tokena prema svim patchovima
            attention_to_cls = feat[0, 0, 1:].cpu().numpy()
        else:
            # Ako je samo feature mapa
            attention_to_cls = feat[0, 1:, :].mean(axis=1).cpu().numpy()

        # Reshape u grid
        grid_size = int(np.sqrt(len(attention_to_cls)))
        if grid_size * grid_size == len(attention_to_cls):
            importance_map = attention_to_cls.reshape(grid_size, grid_size)
        else:
            importance_map = attention_to_cls[:196].reshape(14, 14)

        # Normalizacija i skaliranje
        importance_map = (importance_map - importance_map.min()) / (
            importance_map.max() - importance_map.min() + 1e-8
        )

        # Resize i toplinska mapa
        importance_map = cv2.resize(importance_map, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * importance_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Preklapanje
        img_np = np.array(img_224)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        # Base64
        res_pil = Image.fromarray(overlay)
        buffered = BytesIO()
        res_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _generate_fallback_attention(self, pil_image):
        """Fallback metoda ako hooks ne rade."""
        print("⚠️ Using fallback attention for RAW ViT")
        img_224 = pil_image.resize((224, 224))

        # Jednostavna saliency mapa
        img_np = np.array(img_224).astype(np.float32) / 255.0
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Edge detection kao aproksimacija "attentiona"
        edges = cv2.Canny((gray * 255).astype(np.uint8), 100, 200)
        edges = cv2.resize(edges, (224, 224))

        heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted((img_np * 255).astype(np.uint8), 0.6, heatmap, 0.4, 0)

        res_pil = Image.fromarray(overlay)
        buffered = BytesIO()
        res_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def cleanup(self):
        for h in self.hooks:
            h.remove()
