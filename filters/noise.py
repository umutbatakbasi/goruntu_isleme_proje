import numpy as np
import random


def salt_pepper_noise_manual(img: np.ndarray, noise_ratio: float = 0.05) -> np.ndarray:
    """
    Görüntüye Salt & Pepper gürültüsü ekler.

    noise_ratio:
        Gürültü eklenecek piksel oranı.
        Örn: 0.05 -> piksellerin %5'i değişir.

    Salt  -> 255
    Pepper -> 0
    """

    if not (0 <= noise_ratio <= 1):
        raise ValueError("noise_ratio değeri 0 ile 1 arasında olmalıdır.")

    noisy = img.copy()
    h, w = img.shape[:2]

    total_pixels = h * w
    noisy_pixels = int(total_pixels * noise_ratio)

    for _ in range(noisy_pixels):
        i = random.randint(0, h - 1)
        j = random.randint(0, w - 1)

        noise_value = 255 if random.random() < 0.5 else 0

        if len(img.shape) == 2:
            noisy[i, j] = noise_value
        else:
            # Renkli görüntüde seçilen pikselin tüm kanallarını aynı değere çekiyoruz
            noisy[i, j, 0] = noise_value
            noisy[i, j, 1] = noise_value
            noisy[i, j, 2] = noise_value

    return noisy