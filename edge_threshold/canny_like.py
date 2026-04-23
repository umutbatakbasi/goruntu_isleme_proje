import numpy as np

from filters.mean_filter import mean_filter_manual
from edge_threshold.double_threshold import double_threshold_manual


def compute_gradient_manual(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gri seviyeli görüntü için manuel gradyan hesaplar.
    Basit türev maskeleri kullanılır.

    Gx:
    [-1  0  1]
    [-1  0  1]
    [-1  0  1]

    Gy:
    [-1 -1 -1]
    [ 0  0  0]
    [ 1  1  1]
    """

    if len(img.shape) != 2:
        raise ValueError("Gradient hesabı için gri seviyeli görüntü gerekli.")

    h, w = img.shape

    gx = np.zeros((h, w), dtype=np.float64)
    gy = np.zeros((h, w), dtype=np.float64)
    magnitude = np.zeros((h, w), dtype=np.float64)

    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float64)

    kernel_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float64)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            sum_x = 0.0
            sum_y = 0.0

            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    pixel = float(img[i + ki, j + kj])
                    sum_x += pixel * kernel_x[ki + 1, kj + 1]
                    sum_y += pixel * kernel_y[ki + 1, kj + 1]

            gx[i, j] = sum_x
            gy[i, j] = sum_y
            magnitude[i, j] = np.sqrt(sum_x ** 2 + sum_y ** 2)

    return gx, gy, magnitude


def normalize_to_255_manual(img: np.ndarray) -> np.ndarray:
    """
    Float görüntüyü 0-255 aralığına normalize eder.
    """

    min_val = float(np.min(img))
    max_val = float(np.max(img))

    if max_val == min_val:
        return np.zeros_like(img, dtype=np.uint8)

    h, w = img.shape
    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            value = ((img[i, j] - min_val) * 255.0) / (max_val - min_val)
            if value < 0:
                value = 0
            elif value > 255:
                value = 255
            out[i, j] = int(round(value))

    return out


def edge_tracking_by_hysteresis_manual(
    dt_img: np.ndarray,
    weak_value: int = 128,
    strong_value: int = 255
) -> np.ndarray:
    """
    Zayıf kenarlar güçlü kenara 8-komşuluk içinde bağlıysa güçlü kenar yapılır.
    Değilse sıfırlanır.
    """

    h, w = dt_img.shape
    out = dt_img.copy()

    changed = True
    while changed:
        changed = False

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if out[i, j] == weak_value:
                    has_strong_neighbor = False

                    for ki in range(-1, 2):
                        for kj in range(-1, 2):
                            if ki == 0 and kj == 0:
                                continue
                            if out[i + ki, j + kj] == strong_value:
                                has_strong_neighbor = True
                                break
                        if has_strong_neighbor:
                            break

                    if has_strong_neighbor:
                        out[i, j] = strong_value
                        changed = True

    # Bağlantısız kalan zayıf kenarları sıfırla
    for i in range(h):
        for j in range(w):
            if out[i, j] == weak_value:
                out[i, j] = 0

    return out


def canny_like_manual(
    img: np.ndarray,
    smoothing_kernel_size: int = 3,
    low_threshold: int = 50,
    high_threshold: int = 120,
    weak_value: int = 128,
    strong_value: int = 255
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Canny-benzeri manuel kenar bulma akışı.

    Dönüş:
    - smoothed görüntü
    - gradient magnitude normalize edilmiş görüntü
    - final edge görüntüsü
    """

    if len(img.shape) != 2:
        raise ValueError("Canny-like işlem için gri seviyeli görüntü gerekli.")

    # 1) smoothing
    smoothed = mean_filter_manual(img, kernel_size=smoothing_kernel_size)

    # 2) gradient
    _, _, magnitude = compute_gradient_manual(smoothed)

    # 3) normalize magnitude
    magnitude_norm = normalize_to_255_manual(magnitude)

    # 4) double threshold
    dt = double_threshold_manual(
        magnitude_norm,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        weak_value=weak_value,
        strong_value=strong_value
    )

    # 5) hysteresis benzeri takip
    final_edges = edge_tracking_by_hysteresis_manual(
        dt,
        weak_value=weak_value,
        strong_value=strong_value
    )

    return smoothed, magnitude_norm, final_edges