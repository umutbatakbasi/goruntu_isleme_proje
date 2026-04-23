import numpy as np


def create_structuring_element_manual(size: int = 3, shape: str = "square") -> np.ndarray:
    """
    Yapısal eleman oluşturur.

    shape:
        "square" -> tüm elemanlar 1
        "cross"  -> artı biçimi
    """

    if size % 2 == 0 or size < 1:
        raise ValueError("size pozitif tek sayı olmalıdır.")

    se = np.zeros((size, size), dtype=np.uint8)

    if shape == "square":
        for i in range(size):
            for j in range(size):
                se[i, j] = 1

    elif shape == "cross":
        center = size // 2
        for i in range(size):
            se[i, center] = 1
            se[center, i] = 1

    else:
        raise ValueError("shape yalnızca 'square' veya 'cross' olabilir.")

    return se


def dilate_manual(img: np.ndarray, se_size: int = 3, se_shape: str = "square") -> np.ndarray:
    """
    Binary görüntüye manuel dilation uygular.

    Giriş görüntüsünün 0 ve 255 değerlerinden oluşması beklenir.
    """

    if len(img.shape) != 2:
        raise ValueError("Dilation için gri/binary tek kanallı görüntü gerekli.")

    h, w = img.shape
    pad = se_size // 2
    out = np.zeros((h, w), dtype=np.uint8)

    se = create_structuring_element_manual(se_size, se_shape)

    for i in range(h):
        for j in range(w):
            should_activate = False

            for ki in range(-pad, pad + 1):
                for kj in range(-pad, pad + 1):
                    ni = i + ki
                    nj = j + kj

                    if 0 <= ni < h and 0 <= nj < w:
                        if se[ki + pad, kj + pad] == 1 and img[ni, nj] == 255:
                            should_activate = True
                            break
                if should_activate:
                    break

            out[i, j] = 255 if should_activate else 0

    return out