import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from typing import Dict, Tuple, List, Optional
import asyncio


class InferenceService:
    """Servis za inferencu sa optimizacijama za batch"""

    def __init__(self, models: Dict[str, torch.nn.Module]):
        self.models = models
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.labels = ["AI", "Real"]

        # Precompile models ako je moguće
        for name, model in self.models.items():
            if model is not None:
                try:
                    model.eval()
                    # Warmup model
                    with torch.no_grad():
                        dummy_input = torch.randn(1, 3, 224, 224)
                        _ = model(dummy_input)
                except:
                    pass

    def preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        return self.transform(pil_image).unsqueeze(0)

    def predict_single(
        self, model_name: str, input_tensor: torch.Tensor, device: torch.device
    ) -> Dict:
        if model_name not in self.models or self.models[model_name] is None:
            return {
                "error": f"Model {model_name} not available",
                "class_idx": 0,
                "confidence": 0.0,
                "probabilities": [0.5, 0.5],
            }

        model = self.models[model_name]
        try:
            with torch.no_grad():
                model.eval()
                output = model(input_tensor.to(device))
                probabilities = F.softmax(output, dim=1)[0]
                confidence, class_idx = torch.max(probabilities, dim=0)

                return {
                    "output": output,
                    "class_idx": class_idx.item(),
                    "confidence": confidence.item(),
                    "probabilities": probabilities.cpu().tolist(),
                    "label": self.labels[class_idx.item()],
                    "error": None,
                }
        except Exception as e:
            print(f"❌ Prediction error for {model_name}: {e}")
            return {
                "error": str(e),
                "class_idx": 0,
                "confidence": 0.0,
                "probabilities": [0.5, 0.5],
            }

    def predict_batch(
        self, input_tensors: List[torch.Tensor], device: torch.device
    ) -> Dict[str, Dict]:
        """OPTIMIZIRANO: Batch predikcije sa paralelnim modelima."""
        if not input_tensors:
            return {}

        batch_tensor = torch.cat(input_tensors, dim=0)
        batch_size = batch_tensor.shape[0]
        results = {}

        for model_name, model in self.models.items():
            if model is None:
                continue

            try:
                with torch.no_grad():
                    model.eval()
                    # Koristi veći batch size za effikasnije GPU korištenje
                    outputs = model(batch_tensor.to(device))
                    probs = F.softmax(outputs, dim=1)
                    confidences, class_idxs = torch.max(probs, dim=1)

                    model_results = []
                    for i in range(batch_size):
                        model_results.append(
                            {
                                "class_idx": class_idxs[i].item(),
                                "confidence": confidences[i].item(),
                                "probabilities": probs[i].cpu().tolist(),
                                "label": self.labels[class_idxs[i].item()],
                                "error": None,
                            }
                        )

                    results[model_name] = model_results

            except Exception as e:
                print(f"❌ Batch prediction error for {model_name}: {e}")
                results[model_name] = [
                    {
                        "class_idx": 0,
                        "confidence": 0.0,
                        "probabilities": [0.5, 0.5],
                        "label": "AI",
                        "error": str(e),
                    }
                    for _ in range(batch_size)
                ]

        return results

    def predict_all(
        self, input_tensor: torch.Tensor, device: torch.device
    ) -> Dict[str, Dict]:
        results = {}
        for model_name in self.models:
            if self.models[model_name] is not None:
                results[model_name] = self.predict_single(
                    model_name, input_tensor, device
                )
        return results

    def predict_selected(
        self, model_names: List[str], input_tensor: torch.Tensor, device: torch.device
    ) -> Dict[str, Dict]:
        results = {}
        for model_name in model_names:
            if model_name in self.models and self.models[model_name] is not None:
                results[model_name] = self.predict_single(
                    model_name, input_tensor, device
                )
            else:
                results[model_name] = {
                    "error": f"Model {model_name} not available",
                    "class_idx": 0,
                    "confidence": 0.0,
                    "probabilities": [0.5, 0.5],
                }
        return results

    # --- NOVE OPTIMIZIRANE ASYNC METODE ---

    async def process_all_models_parallel(
        self, input_tensors: List[torch.Tensor], device: torch.device, executor
    ) -> List[Dict]:
        """Pokreni SVE modele paralelno za SVE slike (najveća optimizacija)."""
        loop = asyncio.get_event_loop()

        # Ako imamo više slika, koristi batch mode
        if len(input_tensors) > 1:
            return [self.predict_batch(input_tensors, device)]

        # Za jednu sliku pokreni sve modele paralelno
        tasks = []
        model_names = []

        for model_name, model in self.models.items():
            if model is not None:
                tasks.append(
                    loop.run_in_executor(
                        executor, lambda m=model, t=input_tensors[0].to(device): m(t)
                    )
                )
                model_names.append(model_name)

        # Pokreni sve modele paralelno
        try:
            outputs = await asyncio.gather(*tasks, return_exceptions=True)

            # Pripremi rezultate
            all_predictions = {}
            for idx, (model_name, output) in enumerate(zip(model_names, outputs)):
                if isinstance(output, Exception):
                    all_predictions[model_name] = {
                        "class_idx": 0,
                        "confidence": 0.0,
                        "probabilities": [0.5, 0.5],
                        "label": "AI",
                        "error": str(output),
                    }
                else:
                    with torch.no_grad():
                        probs = F.softmax(output, dim=1)[0]
                        conf, class_idx = torch.max(probs, dim=0)

                        all_predictions[model_name] = {
                            "class_idx": class_idx.item(),
                            "confidence": conf.item(),
                            "probabilities": probs.cpu().tolist(),
                            "label": self.labels[class_idx.item()],
                            "error": None,
                        }

            return [all_predictions]

        except Exception as e:
            print(f"❌ Parallel model processing error: {e}")
            return [{}]

    # Originalne async metode (ostaju za backward compatibility)
    async def process_cnn_fine_tuned_prediction(self, input_tensor, device, executor):
        loop = asyncio.get_event_loop()
        with torch.no_grad():
            cnn_out = await loop.run_in_executor(
                executor, lambda: self.models["cnn_fine_tuned"](input_tensor.to(device))
            )
            cnn_probs = F.softmax(cnn_out, dim=1)[0]
            cnn_conf, cnn_class = torch.max(cnn_probs, dim=0)
            return {
                "output": cnn_out,
                "probs": cnn_probs,
                "confidence": cnn_conf.item(),
                "class_idx": cnn_class.item(),
                "probabilities": cnn_probs.cpu().tolist(),
            }

    async def process_cnn_raw_prediction(self, input_tensor, device, executor):
        if self.models.get("cnn_raw") is None:
            return {
                "class_idx": 0,
                "confidence": 0.5,
                "probabilities": [0.5, 0.5],
                "error": "CNN Raw model not loaded",
            }

        loop = asyncio.get_event_loop()
        with torch.no_grad():
            cnn_raw_out = await loop.run_in_executor(
                executor, lambda: self.models["cnn_raw"](input_tensor.to(device))
            )
            cnn_raw_probs = F.softmax(cnn_raw_out, dim=1)[0]
            cnn_raw_conf, cnn_raw_class = torch.max(cnn_raw_probs, dim=0)
            return {
                "output": cnn_raw_out,
                "probs": cnn_raw_probs,
                "confidence": cnn_raw_conf.item(),
                "class_idx": cnn_raw_class.item(),
                "probabilities": cnn_raw_probs.cpu().tolist(),
            }

    async def process_vit_fine_tuned_prediction(self, input_tensor, device, executor):
        loop = asyncio.get_event_loop()
        with torch.no_grad():
            try:
                vit_out = await loop.run_in_executor(
                    executor,
                    lambda: self.models["vit_fine_tuned"](input_tensor.to(device)),
                )
                vit_probs = F.softmax(vit_out, dim=1)[0]
                vit_conf, vit_class = torch.max(vit_probs, dim=0)
                return {
                    "output": vit_out,
                    "probs": vit_probs,
                    "confidence": vit_conf.item(),
                    "class_idx": vit_class.item(),
                    "probabilities": vit_probs.cpu().tolist(),
                    "error": None,
                }
            except Exception as e:
                print(f"ViT fine-tuned prediction error: {e}")
                return {
                    "output": None,
                    "probs": torch.tensor([0.5, 0.5]),
                    "confidence": 0.5,
                    "class_idx": 0,
                    "probabilities": [0.5, 0.5],
                    "error": str(e),
                }

    async def process_vit_raw_prediction(self, input_tensor, device, executor):
        if self.models.get("vit_raw") is None:
            return {
                "class_idx": 0,
                "confidence": 0.5,
                "probabilities": [0.5, 0.5],
                "error": "ViT Raw model not loaded",
            }

        loop = asyncio.get_event_loop()
        with torch.no_grad():
            try:
                vit_raw_out = await loop.run_in_executor(
                    executor, lambda: self.models["vit_raw"](input_tensor.to(device))
                )
                vit_raw_probs = F.softmax(vit_raw_out, dim=1)[0]
                vit_raw_conf, vit_raw_class = torch.max(vit_raw_probs, dim=0)
                return {
                    "output": vit_raw_out,
                    "probs": vit_raw_probs,
                    "confidence": vit_raw_conf.item(),
                    "class_idx": vit_raw_class.item(),
                    "probabilities": vit_raw_probs.cpu().tolist(),
                    "error": None,
                }
            except Exception as e:
                print(f"ViT Raw prediction error: {e}")
                return {
                    "output": None,
                    "probs": torch.tensor([0.5, 0.5]),
                    "confidence": 0.5,
                    "class_idx": 0,
                    "probabilities": [0.5, 0.5],
                    "error": str(e),
                }

    async def process_deit_fine_tuned_prediction(self, input_tensor, device, executor):
        if self.models.get("deit_fine_tuned") is None:
            return {
                "class_idx": 0,
                "confidence": 0.5,
                "probabilities": [0.5, 0.5],
                "error": "DeiT model not loaded",
            }

        loop = asyncio.get_event_loop()
        with torch.no_grad():
            try:
                deit_out = await loop.run_in_executor(
                    executor,
                    lambda: self.models["deit_fine_tuned"](input_tensor.to(device)),
                )
                deit_probs = F.softmax(deit_out, dim=1)[0]
                deit_conf, deit_class = torch.max(deit_probs, dim=0)
                return {
                    "output": deit_out,
                    "probs": deit_probs,
                    "confidence": deit_conf.item(),
                    "class_idx": deit_class.item(),
                    "probabilities": deit_probs.cpu().tolist(),
                    "error": None,
                }
            except Exception as e:
                print(f"DeiT fine-tuned prediction error: {e}")
                return {
                    "output": None,
                    "probs": torch.tensor([0.5, 0.5]),
                    "confidence": 0.5,
                    "class_idx": 0,
                    "probabilities": [0.5, 0.5],
                    "error": str(e),
                }

    async def process_convnext_fine_tuned_prediction(
        self, input_tensor, device, executor
    ):
        if self.models.get("convnext_fine_tuned") is None:
            return {
                "class_idx": 0,
                "confidence": 0.5,
                "probabilities": [0.5, 0.5],
                "error": "ConvNeXt model not loaded",
            }

        loop = asyncio.get_event_loop()
        with torch.no_grad():
            try:
                convnext_out = await loop.run_in_executor(
                    executor,
                    lambda: self.models["convnext_fine_tuned"](input_tensor.to(device)),
                )
                convnext_probs = F.softmax(convnext_out, dim=1)[0]
                convnext_conf, convnext_class = torch.max(convnext_probs, dim=0)
                return {
                    "output": convnext_out,
                    "probs": convnext_probs,
                    "confidence": convnext_conf.item(),
                    "class_idx": convnext_class.item(),
                    "probabilities": convnext_probs.cpu().tolist(),
                    "error": None,
                }
            except Exception as e:
                print(f"ConvNeXt fine-tuned prediction error: {e}")
                return {
                    "output": None,
                    "probs": torch.tensor([0.5, 0.5]),
                    "confidence": 0.5,
                    "class_idx": 0,
                    "probabilities": [0.5, 0.5],
                    "error": str(e),
                }
