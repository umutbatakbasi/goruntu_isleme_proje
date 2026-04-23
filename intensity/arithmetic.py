import numpy as np


def _check_same_shape(img1: np.ndarray, img2: np.ndarray) -> None:
    """
    İki görüntünün boyutlarını kontrol eder.
    """
    if img1.shape != img2.shape:
        raise ValueError("İki görüntünün boyutları aynı olmalıdır.")


def image_add_manual(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    İki görüntüyü piksel piksel toplar.
    Sonuç 255'i aşarsa 255'e sabitlenir.

    Toplama
    g(x,y)=f1(x,y)+f2(x,y)

    Ama 255'i geçerse:
    g(x,y)=255
    """
    _check_same_shape(img1, img2)

    h, w = img1.shape[:2]
    out = np.zeros_like(img1, dtype=np.uint8)

    if len(img1.shape) == 2:
        for i in range(h):
            for j in range(w):
                value = int(img1[i, j]) + int(img2[i, j])
                if value > 255:
                    value = 255
                out[i, j] = value
    else:
        c = img1.shape[2]
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    value = int(img1[i, j, k]) + int(img2[i, j, k])
                    if value > 255:
                        value = 255
                    out[i, j, k] = value

    return out


def image_subtract_manual(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    İki görüntüyü piksel piksel çıkarır: img1 - img2
    Sonuç 0'ın altına düşerse 0'a sabitlenir.

    Çıkarma
    g(x,y)=f1(x,y)−f2(x,y)

    Ama 0’ın altına düşerse:

    g(x,y)=0
    """
    _check_same_shape(img1, img2)

    h, w = img1.shape[:2]
    out = np.zeros_like(img1, dtype=np.uint8)

    if len(img1.shape) == 2:
        for i in range(h):
            for j in range(w):
                value = int(img1[i, j]) - int(img2[i, j])
                if value < 0:
                    value = 0
                out[i, j] = value
    else:
        c = img1.shape[2]
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    value = int(img1[i, j, k]) - int(img2[i, j, k])
                    if value < 0:
                        value = 0
                    out[i, j, k] = value

    return out


def image_multiply_manual(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    İki görüntüyü piksel piksel çarpar.
    8 bit aralığına geri dönebilmek için sonuç /255 ile normalize edilir.
    g(x,y)= [ f1(x,y).f2(x,y) ] /255
    """
    _check_same_shape(img1, img2)

    h, w = img1.shape[:2]
    out = np.zeros_like(img1, dtype=np.uint8)

    if len(img1.shape) == 2:
        for i in range(h):
            for j in range(w):
                value = (int(img1[i, j]) * int(img2[i, j])) / 255.0
                if value > 255:
                    value = 255
                out[i, j] = int(round(value))
    else:
        c = img1.shape[2]
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    value = (int(img1[i, j, k]) * int(img2[i, j, k])) / 255.0
                    if value > 255:
                        value = 255
                    out[i, j, k] = int(round(value))

    return out