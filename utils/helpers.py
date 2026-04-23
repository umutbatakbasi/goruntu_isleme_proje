import numpy as np


def clamp_image(img: np.ndarray) -> np.ndarray:
    """
    Piksel değerlerini 0-255 aralığında tutar ve uint8'e çevirir.
    """
    return np.clip(img, 0, 255).astype(np.uint8)


def is_grayscale(img: np.ndarray) -> bool:
    """
    Görüntü gri seviyeli mi kontrol eder.
    """
    return len(img.shape) == 2


def is_rgb(img: np.ndarray) -> bool:
    """
    Görüntü RGB mi kontrol eder.
    """
    return len(img.shape) == 3 and img.shape[2] == 3


def get_image_info(img: np.ndarray) -> dict:
    """
    Görüntü hakkında temel bilgileri döndürür.
    """
    info = {
        "shape": img.shape,
        "dtype": img.dtype,
        "ndim": img.ndim,
        "min": int(np.min(img)),
        "max": int(np.max(img)),
    }

    if len(img.shape) == 2:
        info["type"] = "Grayscale"
        info["height"], info["width"] = img.shape
    elif len(img.shape) == 3:
        info["type"] = "Color"
        info["height"], info["width"], info["channels"] = img.shape
    else:
        info["type"] = "Unknown"

    return info