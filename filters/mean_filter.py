import numpy as np


def mean_filter_manual(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Gri seviyeli görüntüye manuel mean filter uygular.

    kernel_size tek sayı olmalıdır: 3, 5, 7 ...
    """

    if len(img.shape) != 2:
        raise ValueError("Mean filter için gri seviyeli görüntü gerekli.")

    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size pozitif tek sayı olmalıdır.")

    h, w = img.shape
    pad = kernel_size // 2

    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            total = 0
            count = 0

            for ki in range(-pad, pad + 1):
                for kj in range(-pad, pad + 1):
                    ni = i + ki
                    nj = j + kj

                    if 0 <= ni < h and 0 <= nj < w:
                        total += int(img[ni, nj])
                        count += 1

            mean_value = total / count
            out[i, j] = int(round(mean_value))

    return out