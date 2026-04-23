import numpy as np


def flip_horizontal_manual(img: np.ndarray) -> np.ndarray:
    """
    Görüntüyü yatay eksende aynalar.
    Sol-sağ ters çevrilir.
    """
    h, w = img.shape[:2]
    out = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            out[i, w - 1 - j] = img[i, j]

    return out


def flip_vertical_manual(img: np.ndarray) -> np.ndarray:
    """
    Görüntüyü dikey eksende aynalar.
    Üst-alt ters çevrilir.
    """
    h, w = img.shape[:2]
    out = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            out[h - 1 - i, j] = img[i, j]

    return out