import hashlib
from typing import Dict

prediction_cache: Dict[str, dict] = {}
processing_status: Dict[str, dict] = {}


def get_cache_key(file_contents: bytes) -> str:
    """Generiraj cache key od fajla."""
    file_hash = hashlib.md5(file_contents).hexdigest()
    return f"comparison_{file_hash}"


def cleanup_old_cache(max_size: int = 100):
    """Očisti stari cache ako je prevelik."""
    global prediction_cache
    if len(prediction_cache) > max_size:
        oldest_key = next(iter(prediction_cache))
        del prediction_cache[oldest_key]
        print(f"🧹 Očišćen stari cache entry: {oldest_key}")
