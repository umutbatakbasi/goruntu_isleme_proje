import numpy as np


def gray_to_binary_manual(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Gri seviyeli görüntüyü binary görüntüye çevirir.
    threshold: 0-255 arasında eşik değeri
    """

    if len(img.shape) != 2:
        raise ValueError("Binary dönüşüm için gri seviyeli görüntü gerekli.")

    if threshold < 0 or threshold > 255:
        raise ValueError("Threshold değeri 0 ile 255 arasında olmalıdır.")

    h, w = img.shape
    binary = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if int(img[i, j]) >= threshold:
                binary[i, j] = 255
            else:
                binary[i, j] = 0

    return binary