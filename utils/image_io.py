from pathlib import Path
from PIL import Image
import numpy as np


def load_image(path: str) -> np.ndarray:
    """
    Görüntüyü diskten okur ve NumPy dizisi olarak döndürür.
    Hazır görüntü işleme algoritması yapmaz; sadece yükleme işidir.
    """
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Görüntü bulunamadı: {path}")

    img = Image.open(image_path)
    return np.array(img)


def save_image(img: np.ndarray, path: str) -> None:
    """
    NumPy dizisini güvenli şekilde uint8 formata çevirip kaydeder.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("Kaydedilecek veri NumPy dizisi olmalıdır.")

    safe_img = np.clip(img, 0, 255).astype(np.uint8)
    out = Image.fromarray(safe_img)
    out.save(path)