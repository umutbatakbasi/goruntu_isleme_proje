import numpy as np
import math


def rotate_manual(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Görüntüyü verilen açı kadar saat yönünün tersine döndürür.
    Genel açılı manuel döndürme yapılır.
    Yöntem: inverse mapping + nearest neighbor
    """

    h, w = img.shape[:2]
    angle_rad = math.radians(angle_deg)

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Orijinal görüntü merkezi
    cx = w / 2.0
    cy = h / 2.0

    # Köşeleri merkeze göre döndürüp yeni boyutu bul
    corners = [
        (-cx, -cy),
        (w - cx, -cy),
        (-cx, h - cy),
        (w - cx, h - cy)
    ]

    rotated_corners = []
    for x, y in corners:
        xr = x * cos_a - y * sin_a
        yr = x * sin_a + y * cos_a
        rotated_corners.append((xr, yr))

    xs = [p[0] for p in rotated_corners]
    ys = [p[1] for p in rotated_corners]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    new_w = int(math.ceil(max_x - min_x))
    new_h = int(math.ceil(max_y - min_y))

    if len(img.shape) == 2:
        out = np.zeros((new_h, new_w), dtype=img.dtype)
    else:
        out = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)

    # Yeni görüntü merkezi
    new_cx = new_w / 2.0
    new_cy = new_h / 2.0

    # Inverse mapping
    for i in range(new_h):
        for j in range(new_w):
            # Yeni görüntü koordinatını merkeze göre al
            x_new = j - new_cx
            y_new = i - new_cy

            # Ters döndürme uygula
            x_old = x_new * cos_a + y_new * sin_a
            y_old = -x_new * sin_a + y_new * cos_a

            # Eski görüntü koordinat sistemine geri dön
            src_x = int(round(x_old + cx))
            src_y = int(round(y_old + cy))

            if 0 <= src_x < w and 0 <= src_y < h:
                out[i, j] = img[src_y, src_x]

    return out