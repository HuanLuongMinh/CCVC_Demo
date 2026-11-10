"""Readers for OpenEarthMap .tif tiles.

PIL reads the Kaggle OEM tiles directly (RGB uint8 images, 'L' uint8 labels); tifffile is
only a fallback for tiles PIL cannot decode or that come with an unexpected band layout.
"""
import numpy as np
from PIL import Image


def _read_array(path):
    try:
        return np.array(Image.open(path))
    except Exception:  # noqa: BLE001 - PIL cannot decode some GeoTIFF variants
        import tifffile
        return tifffile.imread(path)


def read_rgb(path):
    """Return a PIL RGB uint8 image."""
    try:
        img = Image.open(path)
        if img.mode == 'RGB':
            return img.convert('RGB')
    except Exception:  # noqa: BLE001
        pass
    arr = _read_array(path)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = arr.transpose(1, 2, 0)                    # CHW -> HWC
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr[..., :3]
    if arr.dtype != np.uint8:                           # stretch other dtypes to 0..255
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = ((arr - lo) / max(hi - lo, 1e-6) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode='RGB')


def read_label(path):
    """Return the raw label map as a 2-D uint8 numpy array (OEM values 0..8)."""
    arr = _read_array(path)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    return arr.astype(np.uint8)


def oem_label_to_train(arr):
    """OEM mapping: 0 'unknown' -> 255 (ignored), classes 1..8 -> 0..7."""
    out = arr.astype(np.int16) - 1
    out[out < 0] = 255
    return out.astype(np.uint8)
