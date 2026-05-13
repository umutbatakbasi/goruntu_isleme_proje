import numpy as np

"""
RGB → Gray:
Gray = 0.299 R + 0.587 G + 0.114 B
"""

def rgb_to_gray_manual(img: np.ndarray) -> np.ndarray:
    """
    RGB görüntüyü gri seviyeye çevirir.
    Tamamen manuel (piksel piksel) işlem yapılır.
    """

    # Eğer zaten grayscale ise direkt döndür
    if len(img.shape) == 2:
        return img.copy().astype(np.uint8)

    h, w, c = img.shape

    if c != 3:
        raise ValueError("RGB görüntü bekleniyor!")

    gray = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            r = int(img[i, j, 0]) #img dizisinin 3. elemani kanali belirtir
            g = int(img[i, j, 1])
            b = int(img[i, j, 2])

            # matematiksel dönüşüm
            val = 0.299 * r + 0.587 * g + 0.114 * b

            gray[i, j] = int(val)

    return gray