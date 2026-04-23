from utils.image_io import load_image, save_image
from utils.display import print_image_info
from intensity.grayscale import rgb_to_gray_manual
from intensity.binary import gray_to_binary_manual


def main():
    input_path = "images/lena.png"

    try:
        # 1) Orijinal görüntüyü yükle
        img = load_image(input_path)
        print_image_info(img, "Orijinal Görüntü")

        # 2) Gri dönüşüm
        gray = rgb_to_gray_manual(img)
        print_image_info(gray, "Gri Görüntü")
        save_image(gray, "images/lena_gray.png")

        # 3) Binary dönüşüm
        threshold_value = 128   
        """ 64, 128 ve 180 değerleri için test edilebilir. Böylece eşik değeri değişince görüntünün nasıl değiştiği görülebilir."""
        binary = gray_to_binary_manual(gray, threshold=threshold_value)
        print_image_info(binary, f"Binary Görüntü (Threshold={threshold_value})")
        save_image(binary, "images/lena_binary.png")

        print("Gri ve binary dönüşüm tamamlandı.")

    except FileNotFoundError as e:
        print(f"Hata: {e}")
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")


if __name__ == "__main__":
    main()