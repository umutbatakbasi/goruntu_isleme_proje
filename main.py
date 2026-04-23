from utils.image_io import load_image, save_image
from utils.display import print_image_info
from intensity.grayscale import rgb_to_gray_manual


def main():
    input_path = "images/lena.png"

    img = load_image(input_path)
    print_image_info(img, "Orijinal Görüntü")

    # Gri dönüşüm
    gray = rgb_to_gray_manual(img)
    print_image_info(gray, "Gri Görüntü")

    # kaydet
    save_image(gray, "images/lena_gray.png")

    print("Gri dönüşüm tamamlandı.")


if __name__ == "__main__":
    main()