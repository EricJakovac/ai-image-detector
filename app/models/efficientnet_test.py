import torch
import torch.nn as nn
import timm


class EfficientNetTest(nn.Module):
    """
    EfficientNet-B0 model za binary klasifikaciju (AI vs real).
    Koristi pretrained ImageNet težine.
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # EfficientNet-B0: mali, brz, ima 100% pretrained težine
        self.backbone = timm.create_model(
            'efficientnet_b0',  # ✅ dostupan i testiran
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=3
        )

    def forward(self, x):
        return self.backbone(x)
    
    def load_trained_weights(self, checkpoint_path):
        """Učitaj trenirane težine."""
        self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.eval()  # Inference mode
        print(f"✅ Trenirane težine učitane: {checkpoint_path}")

    @torch.no_grad()
    def predict_test(self, image_tensor):
        """
        Predikcija za jednu sliku (tensor [1, 3, 224, 224]).
        Vraća vjerojatnost da je AI (klasa 0).
        """
        self.eval()
        output = self(image_tensor)
        probs = torch.softmax(output, dim=1)
        ai_prob = probs[0, 0].item()
        return ai_prob
    def load_trained_weights(self, checkpoint_path):
        """Učitaj trenirane težine."""
        self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Učitane težine iz {checkpoint_path}")
        
    @torch.no_grad()
    def batch_predict(self, image_tensors):
        """
        Batch predikcija za više slika [N, 3, 224, 224].
        Vraća listu vjerojatnosti AI.
        """
        self.eval()
        outputs = self(image_tensors)
        probs = torch.softmax(outputs, dim=1)
        ai_probs = probs[:, 0].cpu().numpy().tolist()  # AI class = index 0
        return ai_probs
