import numpy as np


def crop_manual(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Görüntüyü verilen koordinatlara göre kırpar.

    Parametreler:
    x1, y1 -> sol üst köşe
    x2, y2 -> sağ alt köşe

    Not:
    x ekseni sütunları, y ekseni satırları temsil eder.
    """

    h, w = img.shape[:2]

    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise ValueError("Kırpma koordinatları görüntü sınırları dışında.")

    if x1 >= x2 or y1 >= y2:
        raise ValueError("Geçersiz kırpma koordinatları.")

    # Manuel kırpma
    if len(img.shape) == 2:
        out = np.zeros((y2 - y1, x2 - x1), dtype=img.dtype)
    else:
        out = np.zeros((y2 - y1, x2 - x1, img.shape[2]), dtype=img.dtype)

    for i in range(y1, y2):
        for j in range(x1, x2):
            out[i - y1, j - x1] = img[i, j]

    return out