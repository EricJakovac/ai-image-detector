import torch
from PIL import Image
from typing import Optional, Dict, List
import asyncio
import hashlib
from functools import lru_cache

# Import svih XAI generatora
try:
    from xai.cnn_gradcam import GradCamGenerator, GradCamGeneratorRaw

    print("✅ Grad-CAM generatori učitani")
except ImportError:
    print("⚠️ Grad-CAM generatori nisu pronađeni")

try:
    from xai.vit_attention import ViTAttentionGenerator, ViTAttentionGeneratorRaw

    print("✅ ViT Attention generatori učitani")
except ImportError:
    print("⚠️ ViT Attention generatori nisu pronađeni")

try:
    from xai.deit_attention import DeiTAttentionGenerator

    print("✅ DeiT Attention generator učitani")
except ImportError:
    print("⚠️ DeiT Attention generator nije pronađen")

try:
    from xai.convnext_gradcam import ConvNeXtGradCamGenerator

    print("✅ ConvNeXt Grad-CAM generatori učitani")
except ImportError:
    print("⚠️ ConvNeXt Grad-CAM generatori nisu pronađeni")


class XAIService:
    """Centralizirani servis za XAI generiranje sa optimizacijama"""

    def __init__(self, models: Dict[str, torch.nn.Module]):
        self.models = models
        self.xai_generators = {}
        self.xai_cache = {}  # Cache za XAI rezultate
        self._initialize_xai_generators()

    def _initialize_xai_generators(self):
        """Inicijaliziraj sve XAI generatore jednom"""
        print("\n🎨 Inicijalizacija XAI generatora...")

        # 1. CNN Fine-tuned - Grad-CAM
        if (
            "cnn_fine_tuned" in self.models
            and self.models["cnn_fine_tuned"] is not None
        ):
            try:
                model_obj = self.models["cnn_fine_tuned"]
                cnn_model = (
                    model_obj.model if hasattr(model_obj, "model") else model_obj
                )
                self.xai_generators["cnn_fine_tuned"] = GradCamGenerator(cnn_model)
                print("✅ Grad-CAM za CNN fine-tuned")
            except Exception as e:
                print(f"❌ Grad-CAM CNN fine-tuned error: {e}")

        # 2. CNN Raw - Grad-CAM Raw
        if "cnn_raw" in self.models and self.models["cnn_raw"] is not None:
            try:
                self.xai_generators["cnn_raw"] = GradCamGeneratorRaw(
                    self.models["cnn_raw"].model
                )
                print("✅ Grad-CAM za CNN raw")
            except Exception as e:
                print(f"⚠️ Grad-CAM CNN raw error: {e}")

        # 3. ViT Fine-tuned - Attention
        if (
            "vit_fine_tuned" in self.models
            and self.models["vit_fine_tuned"] is not None
        ):
            try:
                vit_model = self.models["vit_fine_tuned"]
                self.xai_generators["vit_fine_tuned"] = ViTAttentionGenerator(
                    vit_model.model
                )
                print("✅ Attention za ViT fine-tuned")
            except Exception as e:
                print(f"⚠️ Attention ViT fine-tuned error: {e}")

        # 4. ViT Raw - Attention Raw
        if "vit_raw" in self.models and self.models["vit_raw"] is not None:
            try:
                vit_raw_model = self.models["vit_raw"]
                self.xai_generators["vit_raw"] = ViTAttentionGeneratorRaw(
                    vit_raw_model.model
                )
                print("✅ Attention za ViT raw")
            except Exception as e:
                print(f"⚠️ Attention ViT raw error: {e}")

        # 5. DeiT Fine-tuned - Attention
        if (
            "deit_fine_tuned" in self.models
            and self.models["deit_fine_tuned"] is not None
        ):
            try:
                deit_model = self.models["deit_fine_tuned"]
                self.xai_generators["deit_fine_tuned"] = DeiTAttentionGenerator(
                    deit_model.model
                )
                print("✅ Attention za DeiT fine-tuned")
            except Exception as e:
                print(f"⚠️ Attention DeiT error: {e}")

        # 6. ConvNeXt Fine-tuned - Grad-CAM
        if (
            "convnext_fine_tuned" in self.models
            and self.models["convnext_fine_tuned"] is not None
        ):
            try:
                self.xai_generators["convnext_fine_tuned"] = ConvNeXtGradCamGenerator(
                    self.models["convnext_fine_tuned"].model
                )
                print("✅ Grad-CAM za ConvNeXt fine-tuned")
            except Exception as e:
                print(f"⚠️ Grad-CAM ConvNeXt error: {e}")

        print(f"✨ {len(self.xai_generators)} XAI generatora inicijalizirano")

    # --- OPTIMIZIRANE ASYNC METODE ---

    async def process_all_xai_parallel(
        self, pil_images: List[Image.Image], predictions_list: List[Dict], executor
    ) -> List[Dict]:
        """Generiraj SVE XAI vizualizacije za SVE slike paralelno (najveća optimizacija)."""
        loop = asyncio.get_event_loop()
        all_xai_results = []

        for img_idx, (pil_image, predictions) in enumerate(
            zip(pil_images, predictions_list)
        ):
            xai_results = {}
            tasks = []

            # CNN Fine-tuned
            if (
                "cnn_fine_tuned" in predictions
                and "cnn_fine_tuned" in self.xai_generators
            ):
                tasks.append(
                    (
                        "cnn_fine_tuned",
                        loop.run_in_executor(
                            executor,
                            lambda img=pil_image, tc=predictions["cnn_fine_tuned"][
                                "class_idx"
                            ]: self.xai_generators["cnn_fine_tuned"].generate_single(
                                img, tc
                            ),
                        ),
                    )
                )

            # CNN Raw
            if "cnn_raw" in predictions and "cnn_raw" in self.xai_generators:
                tasks.append(
                    (
                        "cnn_raw",
                        loop.run_in_executor(
                            executor,
                            lambda img=pil_image, tc=predictions["cnn_raw"][
                                "class_idx"
                            ]: self.xai_generators["cnn_raw"].generate_single(img, tc),
                        ),
                    )
                )

            # ViT Fine-tuned
            if (
                "vit_fine_tuned" in predictions
                and "vit_fine_tuned" in self.xai_generators
            ):
                tasks.append(
                    (
                        "vit_fine_tuned",
                        loop.run_in_executor(
                            executor,
                            lambda img=pil_image: self.xai_generators[
                                "vit_fine_tuned"
                            ].generate_attention_map(img),
                        ),
                    )
                )

            # ViT Raw
            if "vit_raw" in predictions and "vit_raw" in self.xai_generators:
                tasks.append(
                    (
                        "vit_raw",
                        loop.run_in_executor(
                            executor,
                            lambda img=pil_image: self.xai_generators[
                                "vit_raw"
                            ].generate_attention_map(img),
                        ),
                    )
                )

            # DeiT Fine-tuned
            if (
                "deit_fine_tuned" in predictions
                and "deit_fine_tuned" in self.xai_generators
            ):
                tasks.append(
                    (
                        "deit_fine_tuned",
                        loop.run_in_executor(
                            executor,
                            lambda img=pil_image: self.xai_generators[
                                "deit_fine_tuned"
                            ].generate_attention_map(img),
                        ),
                    )
                )

            # ConvNeXt Fine-tuned
            if (
                "convnext_fine_tuned" in predictions
                and "convnext_fine_tuned" in self.xai_generators
            ):
                tasks.append(
                    (
                        "convnext_fine_tuned",
                        loop.run_in_executor(
                            executor,
                            lambda img=pil_image, tc=predictions["convnext_fine_tuned"][
                                "class_idx"
                            ]: self.xai_generators[
                                "convnext_fine_tuned"
                            ].generate_single(
                                img, tc
                            ),
                        ),
                    )
                )

            # Pokreni sve XAI zadatke paralelno za ovu sliku
            try:
                # Pripremi taskove
                xai_tasks = [task for _, task in tasks]

                # Pokreni sve paralelno
                results = await asyncio.gather(*xai_tasks, return_exceptions=True)

                # Mapiraj rezultate
                for idx, (model_name, _) in enumerate(tasks):
                    if idx < len(results) and not isinstance(results[idx], Exception):
                        xai_results[model_name] = results[idx]
                    else:
                        xai_results[model_name] = None

            except Exception as e:
                print(f"⚠️ XAI generation failed for image {img_idx}: {e}")

            all_xai_results.append(xai_results)

        return all_xai_results

    # Backward compatibility metode (ostaju iste)
    async def process_gradcam_fine_tuned(self, pil_image, target_class, executor):
        loop = asyncio.get_event_loop()
        try:
            if "cnn_fine_tuned" in self.xai_generators:
                return await loop.run_in_executor(
                    executor,
                    lambda: self.xai_generators["cnn_fine_tuned"].generate_single(
                        pil_image, target_class
                    ),
                )
        except Exception as e:
            print(f"⚠️ Grad-CAM error for fine-tuned: {e}")
        return None

    async def process_gradcam_raw(self, pil_image, target_class, executor):
        loop = asyncio.get_event_loop()
        try:
            if "cnn_raw" in self.xai_generators:
                cnn_cam = await loop.run_in_executor(
                    executor,
                    lambda: self.xai_generators["cnn_raw"].generate_single(
                        pil_image, target_class
                    ),
                )
                return cnn_cam
        except Exception as e:
            print(f"⚠️ Grad-CAM error for raw: {e}")
        return None

    async def process_vit_attention_fine_tuned(self, pil_image, executor):
        loop = asyncio.get_event_loop()
        try:
            if "vit_fine_tuned" in self.xai_generators:
                return await loop.run_in_executor(
                    executor,
                    lambda: self.xai_generators[
                        "vit_fine_tuned"
                    ].generate_attention_map(pil_image),
                )
        except Exception as e:
            print(f"⚠️ ViT attention error for fine-tuned: {e}")
        return None

    async def process_vit_attention_raw(self, pil_image, executor):
        loop = asyncio.get_event_loop()
        try:
            if "vit_raw" in self.xai_generators:
                return await loop.run_in_executor(
                    executor,
                    lambda: self.xai_generators["vit_raw"].generate_attention_map(
                        pil_image
                    ),
                )
        except Exception as e:
            print(f"⚠️ ViT attention error for raw: {e}")
        return None

    async def process_deit_attention_fine_tuned(self, pil_image, executor):
        loop = asyncio.get_event_loop()
        try:
            if "deit_fine_tuned" in self.xai_generators:
                return await loop.run_in_executor(
                    executor,
                    lambda: self.xai_generators[
                        "deit_fine_tuned"
                    ].generate_attention_map(pil_image),
                )
        except Exception as e:
            print(f"⚠️ DeiT attention error for fine-tuned: {e}")
        return None

    async def process_convnext_gradcam_fine_tuned(
        self, pil_image, target_class, executor
    ):
        loop = asyncio.get_event_loop()
        try:
            if "convnext_fine_tuned" in self.xai_generators:
                return await loop.run_in_executor(
                    executor,
                    lambda: self.xai_generators["convnext_fine_tuned"].generate_single(
                        pil_image, target_class
                    ),
                )
        except Exception as e:
            print(f"⚠️ ConvNeXt Grad-CAM error for fine-tuned: {e}")
        return None

    def get_available_generators(self) -> list:
        return list(self.xai_generators.keys())
