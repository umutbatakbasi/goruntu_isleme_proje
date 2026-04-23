import numpy as np


def resize_nn_manual(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Nearest Neighbor yöntemi ile görüntüyü boyutlandırır.

    scale > 1  -> büyütme
    scale < 1  -> küçültme
    """

    if scale <= 0:
        raise ValueError("Scale değeri 0'dan büyük olmalıdır.")

    h, w = img.shape[:2]

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    if len(img.shape) == 2:
        out = np.zeros((new_h, new_w), dtype=img.dtype)
    else:
        out = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)

    for i in range(new_h):
        for j in range(new_w):
            src_i = int(round(i / scale))
            src_j = int(round(j / scale))

            if src_i >= h:
                src_i = h - 1
            if src_j >= w:
                src_j = w - 1

            out[i, j] = img[src_i, src_j]

    return out