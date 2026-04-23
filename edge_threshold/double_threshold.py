import numpy as np


def double_threshold_manual(
    img: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    weak_value: int = 128,
    strong_value: int = 255
) -> np.ndarray:
    """
    Gri seviyeli görüntüye çift eşikleme uygular.

    low_threshold altı      -> 0
    high_threshold üstü     -> strong_value
    aradaki değerler        -> weak_value
    """

    if len(img.shape) != 2:
        raise ValueError("Çift eşikleme için gri seviyeli görüntü gerekli.")

    if not (0 <= low_threshold <= 255 and 0 <= high_threshold <= 255):
        raise ValueError("Threshold değerleri 0 ile 255 arasında olmalıdır.")

    if low_threshold > high_threshold:
        raise ValueError("low_threshold, high_threshold'dan büyük olamaz.")

    h, w = img.shape
    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            pixel = int(img[i, j])

            if pixel >= high_threshold:
                out[i, j] = strong_value
            elif pixel >= low_threshold:
                out[i, j] = weak_value
            else:
                out[i, j] = 0

    return out