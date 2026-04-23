import numpy as np


def create_motion_kernel_manual(kernel_size: int = 9, direction: str = "horizontal") -> np.ndarray:
    """
    Manuel motion blur kernel oluşturur.

    direction:
        "horizontal"
        "vertical"
        "diag_main"   -> sol üstten sağ alta
        "diag_anti"   -> sağ üstten sol alta
    """

    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size pozitif tek sayı olmalıdır.")

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)

    if direction == "horizontal":
        row = kernel_size // 2
        for j in range(kernel_size):
            kernel[row, j] = 1.0

    elif direction == "vertical":
        col = kernel_size // 2
        for i in range(kernel_size):
            kernel[i, col] = 1.0

    elif direction == "diag_main":
        for i in range(kernel_size):
            kernel[i, i] = 1.0

    elif direction == "diag_anti":
        for i in range(kernel_size):
            kernel[i, kernel_size - 1 - i] = 1.0

    else:
        raise ValueError("direction geçersiz. horizontal, vertical, diag_main, diag_anti kullanın.")

    kernel_sum = np.sum(kernel)
    if kernel_sum != 0:
        kernel = kernel / kernel_sum

    return kernel


def motion_filter_manual(img: np.ndarray, kernel_size: int = 9, direction: str = "horizontal") -> np.ndarray:
    """
    Gri seviyeli görüntüye manuel motion blur uygular.
    """

    if len(img.shape) != 2:
        raise ValueError("Motion filter için gri seviyeli görüntü gerekli.")

    kernel = create_motion_kernel_manual(kernel_size, direction)

    h, w = img.shape
    pad = kernel_size // 2
    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            total = 0.0

            for ki in range(-pad, pad + 1):
                for kj in range(-pad, pad + 1):
                    ni = i + ki
                    nj = j + kj

                    if 0 <= ni < h and 0 <= nj < w:
                        pixel = float(img[ni, nj])
                        weight = kernel[ki + pad, kj + pad]
                        total += pixel * weight

            if total < 0:
                total = 0
            elif total > 255:
                total = 255

            out[i, j] = int(round(total))

    return out