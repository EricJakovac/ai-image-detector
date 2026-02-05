from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
import io
import time
import hashlib
from PIL import Image
import asyncio
import torch

router = APIRouter()

from utils.image_utils import validate_image
from utils.cache_utils import (
    get_cache_key,
    prediction_cache,
    processing_status,
    cleanup_old_cache,
)


async def cleanup_status(request_id: str):
    await asyncio.sleep(300)
    if request_id in processing_status:
        del processing_status[request_id]


def prepare_model_result(result, heatmap, model_type, model_name):
    ai_prob = result.get("probabilities", [0.5, 0.5])[0]
    real_prob = result.get("probabilities", [0.5, 0.5])[1]
    label = result.get("label", "AI")

    model_data = {
        "label": label,
        "probability": round(result.get("confidence", 0.0), 4),
        "confidence_percent": round(result.get("confidence", 0.0) * 100, 1),
        "raw_probabilities": {"ai": round(ai_prob, 4), "real": round(real_prob, 4)},
        "type": model_type,
        "name": model_name,
    }

    if heatmap:
        model_data["visualization"] = heatmap

    return model_data


@router.post("/predict-comparison")
async def predict_comparison(
    request: Request,
    files: list[UploadFile] = File(None),
    file: UploadFile = File(None),
    background_tasks: BackgroundTasks = None,
):
    """OPTIMIZIRANO: Jedinstveni endpoint sa maksimalnom paralelizacijom."""

    # Prikupi sve slike
    all_files = []
    if file:
        all_files.append(file)
    if files:
        all_files.extend(files)

    if len(all_files) == 0:
        raise HTTPException(400, "No images provided")

    if len(all_files) > 3:
        raise HTTPException(400, "Maximum 3 images allowed")

    # Generiraj ID
    if len(all_files) == 1:
        request_id = f"comp_{time.time()}_{hashlib.md5(all_files[0].filename.encode()).hexdigest()[:8]}"
    else:
        request_id = f"batch_{time.time()}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

    processing_status[request_id] = {
        "status": "processing",
        "progress": 0,
        "message": f"Processing {len(all_files)} image{'s' if len(all_files) > 1 else ''}...",
        "filenames": [f.filename for f in all_files],
        "start_time": time.time(),
        "total_images": len(all_files),
    }

    try:
        # Inicijalizacija servisa
        inference_service = request.state.inference_service
        xai_service = request.state.xai_service
        transform = request.state.transform
        device = request.state.device
        executor = request.state.executor

        if not inference_service or not xai_service:
            raise HTTPException(500, "Services not initialized")

        # 1. UČITAJ I PREPROCESSUJ SVE SLIKE ODJEDNOM
        processing_status[request_id].update(
            {"progress": 10, "message": "Loading images..."}
        )

        pil_images = []
        input_tensors = []
        file_infos = []

        for idx, uploaded_file in enumerate(all_files):
            try:
                file_ext = validate_image(uploaded_file)
                contents = await uploaded_file.read()
                pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
                pil_images.append(pil_image)

                input_tensor = transform(pil_image).unsqueeze(0)
                input_tensors.append(input_tensor)

                file_infos.append(
                    {
                        "filename": uploaded_file.filename,
                        "file_type": file_ext,
                        "original_size": pil_image.size,
                        "index": idx,
                    }
                )

            except Exception as e:
                print(f"❌ Error loading image {uploaded_file.filename}: {e}")
                pil_images.append(Image.new("RGB", (224, 224), color="gray"))
                input_tensors.append(torch.zeros(1, 3, 224, 224))
                file_infos.append(
                    {
                        "filename": uploaded_file.filename,
                        "file_type": "unknown",
                        "original_size": (224, 224),
                        "index": idx,
                        "error": str(e),
                    }
                )

        # 2. OPTIMIZIRANO: Pokreni SVE MODELE paralelno za SVE SLIKE
        processing_status[request_id].update(
            {"progress": 30, "message": "Running AI models..."}
        )

        # Koristi optimiziranu metodu za paralelne predikcije
        all_predictions_list = await inference_service.process_all_models_parallel(
            input_tensors, device, executor
        )

        # Ako je batch mode, razdvoji rezultate po slikama
        if len(all_files) > 1 and len(all_predictions_list) == 1:
            # Rezultati su u batch formatu, razdvoji ih
            batch_predictions = all_predictions_list[0]
            all_predictions = []

            for i in range(len(pil_images)):
                img_predictions = {}
                for model_name in batch_predictions:
                    if i < len(batch_predictions[model_name]):
                        img_predictions[model_name] = batch_predictions[model_name][i]
                    else:
                        img_predictions[model_name] = {
                            "class_idx": 0,
                            "confidence": 0.0,
                            "probabilities": [0.5, 0.5],
                            "label": "AI",
                            "error": "Result not found",
                        }
                all_predictions.append(img_predictions)
        else:
            all_predictions = all_predictions_list

        # 3. OPTIMIZIRANO: Generiraj SVE XAI VIZUALIZACIJE paralelno
        processing_status[request_id].update(
            {"progress": 60, "message": "Generating visualizations..."}
        )

        all_xai_results = await xai_service.process_all_xai_parallel(
            pil_images, all_predictions, executor
        )

        # 4. FORMATIRAJ REZULTATE
        processing_status[request_id].update(
            {"progress": 90, "message": "Preparing results..."}
        )

        labels = ["AI", "Real"]
        final_results = []

        for i, (file_info, predictions, xai_results) in enumerate(
            zip(file_infos, all_predictions, all_xai_results)
        ):
            # Izračunaj improvement
            cnn_same = False
            vit_same = False
            cnn_improvement = 0
            vit_improvement = 0

            if "cnn_fine_tuned" in predictions and "cnn_raw" in predictions:
                cnn_same = predictions["cnn_fine_tuned"].get(
                    "class_idx", 0
                ) == predictions["cnn_raw"].get("class_idx", 0)
                if cnn_same:
                    cnn_improvement = (
                        predictions["cnn_fine_tuned"].get("confidence", 0.0)
                        - predictions["cnn_raw"].get("confidence", 0.0)
                    ) * 100

            if "vit_fine_tuned" in predictions and "vit_raw" in predictions:
                vit_same = predictions["vit_fine_tuned"].get(
                    "class_idx", 0
                ) == predictions["vit_raw"].get("class_idx", 0)
                if vit_same:
                    vit_improvement = (
                        predictions["vit_fine_tuned"].get("confidence", 0.0)
                        - predictions["vit_raw"].get("confidence", 0.0)
                    ) * 100

            # Summary
            if cnn_same and vit_same:
                summary = f"Fine-tuning improves CNN by {cnn_improvement:.1f}% and ViT by {vit_improvement:.1f}%"
            elif cnn_same and not vit_same:
                summary = f"Fine-tuning improves CNN by {cnn_improvement:.1f}% and changes ViT prediction"
            elif not cnn_same and vit_same:
                summary = f"Fine-tuning changes CNN prediction and improves ViT by {vit_improvement:.1f}%"
            else:
                summary = "Fine-tuning changes predictions for both models"

            # Pripremi rezultat
            result = {
                "filename": file_info["filename"],
                "file_type": file_info.get("file_type", "jpg"),
                "image_size": f"{file_info['original_size'][0]}x{file_info['original_size'][1]}",
                "processed_size": "224x224",
                "request_id": f"{request_id}_{i}",
                "processing_time": round(
                    time.time() - processing_status[request_id]["start_time"], 2
                ),
                "models": {
                    "cnn_fine_tuned": prepare_model_result(
                        predictions.get("cnn_fine_tuned", {}),
                        xai_results.get("cnn_fine_tuned"),
                        "fine_tuned",
                        "EfficientNet",
                    ),
                    "convnext_fine_tuned": prepare_model_result(
                        predictions.get("convnext_fine_tuned", {}),
                        xai_results.get("convnext_fine_tuned"),
                        "fine_tuned",
                        "ConvNeXt",
                    ),
                    "vit_fine_tuned": prepare_model_result(
                        predictions.get("vit_fine_tuned", {}),
                        xai_results.get("vit_fine_tuned"),
                        "fine_tuned",
                        "Vision Transformer",
                    ),
                    "deit_fine_tuned": prepare_model_result(
                        predictions.get("deit_fine_tuned", {}),
                        xai_results.get("deit_fine_tuned"),
                        "fine_tuned",
                        "DeiT Transformer",
                    ),
                    "cnn_raw": prepare_model_result(
                        predictions.get("cnn_raw", {}),
                        xai_results.get("cnn_raw"),
                        "raw",
                        "EfficientNet (Raw)",
                    ),
                    "vit_raw": prepare_model_result(
                        predictions.get("vit_raw", {}),
                        xai_results.get("vit_raw"),
                        "raw",
                        "Vision Transformer (Raw)",
                    ),
                },
                "comparison": {
                    "cnn_improvement": round(cnn_improvement, 1),
                    "vit_improvement": round(vit_improvement, 1),
                    "cnn_same_prediction": cnn_same,
                    "vit_same_prediction": vit_same,
                    "cnn_prediction_change": (
                        {
                            "from": labels[
                                predictions.get("cnn_raw", {}).get("class_idx", 0)
                            ],
                            "to": labels[
                                predictions.get("cnn_fine_tuned", {}).get(
                                    "class_idx", 0
                                )
                            ],
                        }
                        if not cnn_same
                        else None
                    ),
                    "vit_prediction_change": (
                        {
                            "from": labels[
                                predictions.get("vit_raw", {}).get("class_idx", 0)
                            ],
                            "to": labels[
                                predictions.get("vit_fine_tuned", {}).get(
                                    "class_idx", 0
                                )
                            ],
                        }
                        if not vit_same
                        else None
                    ),
                    "summary": summary,
                },
            }

            final_results.append(result)

        # Cache za single slike
        if len(all_files) == 1:
            contents = await all_files[0].read()
            cache_key = get_cache_key(contents)
            prediction_cache[cache_key] = final_results[0]
            cleanup_old_cache(100)

        # Finalni status
        processing_status[request_id].update(
            {
                "status": "complete",
                "progress": 100,
                "message": f"Processed {len(all_files)} image{'s' if len(all_files) > 1 else ''}",
                "results_count": len(final_results),
            }
        )

        if background_tasks and len(all_files) == 1:
            background_tasks.add_task(cleanup_status, request_id)

        # Vrati rezultate
        if len(final_results) == 1:
            return final_results[0]
        else:
            return {
                "batch_id": request_id,
                "total_images": len(final_results),
                "results": final_results,
            }

    except Exception as e:
        error_msg = str(e)
        if request_id in processing_status:
            processing_status[request_id].update(
                {"status": "error", "message": error_msg}
            )
        raise HTTPException(status_code=500, detail=error_msg)
