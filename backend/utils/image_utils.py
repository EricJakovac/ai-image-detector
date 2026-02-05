import os
from fastapi import HTTPException
from PIL import Image, ImageFile

# Postavke za robusnije učitavanje slika
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


def validate_image(file) -> str:
    """Validiraj uploadanu sliku."""
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"File type {file_ext} not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Provjeri veličinu
    try:
        file.file.seek(0, 2)  # Idi na kraj
        file_size = file.file.tell()
        file.file.seek(0)  # Vrati se na početak

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                400, f"File too large. Max size: {MAX_FILE_SIZE/(1024*1024)}MB"
            )
    except:
        pass  # Ako ne možemo provjeriti veličinu, preskočimo

    return file_ext
