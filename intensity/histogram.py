import numpy as np
import matplotlib.pyplot as plt


def histogram_manual(img: np.ndarray) -> np.ndarray:
    """
    Gri seviyeli görüntünün histogramını manuel hesaplar.
    Çıktı: 256 elemanlı dizi
    """

    if len(img.shape) != 2:
        raise ValueError("Histogram hesabı için gri seviyeli görüntü gerekli.")

    hist = np.zeros(256, dtype=int)
    h, w = img.shape

    for i in range(h):
        for j in range(w):
            pixel_value = int(img[i, j])
            hist[pixel_value] += 1

    return hist


def print_histogram_summary(hist: np.ndarray) -> None:
    """
    Histogram hakkında kısa özet yazdırır.
    """
    total_pixels = int(np.sum(hist))
    nonzero_bins = int(np.count_nonzero(hist))
    max_bin = int(np.argmax(hist))
    max_count = int(np.max(hist))

    print("\n" + "=" * 40)
    print("Histogram Özeti")
    print("=" * 40)
    print(f"Toplam piksel sayısı: {total_pixels}")
    print(f"Sıfır olmayan bin sayısı: {nonzero_bins}")
    print(f"En yoğun gri seviye: {max_bin}")
    print(f"O seviyedeki piksel sayısı: {max_count}")
    print("=" * 40 + "\n")


def plot_histogram(hist: np.ndarray, title: str = "Histogram") -> None:
    """
    Histogramı çizer.
    """
    plt.figure(figsize=(10, 4))
    plt.bar(range(256), hist, width=1.0)
    plt.title(title)
    plt.xlabel("Gri Seviye")
    plt.ylabel("Piksel Sayısı")
    plt.xlim(0, 255)
    plt.tight_layout()
    plt.show()