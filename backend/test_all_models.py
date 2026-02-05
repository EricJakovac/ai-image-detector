# test_all_models_fixed.py
import torch
import sys
import os

# Dodaj backend u path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.model_loader import initialize_all_models


def test_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 Testing on device: {device}")

    try:
        # Pokušaj učitati sve modele
        models = initialize_all_models(
            device=device,
            is_hf_spaces=False,
            repo_id="EricJakovac/ai-image-detector-model",
            hf_token=None,
        )

        print(f"\n{'='*60}")
        print("📋 MODEL LOADING SUMMARY")
        print(f"{'='*60}")

        for name, model in models.items():
            if model is not None:
                # Provjeri koliko parametara je učítano
                total_params = sum(p.numel() for p in model.parameters())
                non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
                percentage = 100.0 * non_zero_params / total_params

                status = "✅" if percentage > 95 else "⚠️" if percentage > 70 else "❌"
                print(
                    f"{status} {name:20s}: {percentage:6.1f}% loaded ({non_zero_params:,}/{total_params:,})"
                )

                # Test predikcije
                model.eval()
                with torch.no_grad():
                    test_input = torch.randn(1, 3, 224, 224).to(device)
                    output = model(test_input)
                    probs = torch.softmax(output, dim=1)
                    ai_prob = probs[0, 1].item()
                    print(f"   Test prediction: AI={ai_prob:.3f}, Real={1-ai_prob:.3f}")
            else:
                print(f"❌ {name:20s}: FAILED TO LOAD")

        print(
            f"\n🎯 Total models loaded: {sum(1 for m in models.values() if m is not None)}/{len(models)}"
        )

    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_all_models()
