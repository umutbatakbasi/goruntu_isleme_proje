import numpy as np


def histogram_stretch_manual(img: np.ndarray) -> np.ndarray:
    """
    Gri seviyeli görüntüde histogram germe (contrast stretching) uygular.
    Formül:
        s = ((r - r_min) * 255) / (r_max - r_min)
    """

    if len(img.shape) != 2:
        raise ValueError("Histogram germe için gri seviyeli görüntü gerekli.")

    h, w = img.shape
    out = np.zeros((h, w), dtype=np.uint8)

    r_min = int(np.min(img))
    r_max = int(np.max(img))

    if r_max == r_min:
        return img.copy().astype(np.uint8)

    for i in range(h):
        for j in range(w):
            r = int(img[i, j])
            s = ((r - r_min) * 255) / (r_max - r_min)
            out[i, j] = int(round(s))

    return out


def contrast_reduce_manual(img: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """
    Kontrast azaltma işlemi yapar.
    factor 0 ile 1 arasında olmalıdır.
    1'e yaklaştıkça kontrast daha az azalır.
    0'a yaklaştıkça görüntü orta griye yaklaşır.
    """

    if len(img.shape) != 2:
        raise ValueError("Kontrast azaltma için gri seviyeli görüntü gerekli.")

    if not (0 <= factor <= 1):
        raise ValueError("factor değeri 0 ile 1 arasında olmalıdır.")

    h, w = img.shape
    out = np.zeros((h, w), dtype=np.uint8)

    mean_gray = 128

    for i in range(h):
        for j in range(w):
            r = int(img[i, j])
            s = mean_gray + factor * (r - mean_gray)

            if s < 0:
                s = 0
            elif s > 255:
                s = 255

            out[i, j] = int(round(s))

    return out