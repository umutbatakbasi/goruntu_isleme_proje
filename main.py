from utils.image_io import load_image, save_image
from utils.display import print_image_info


def main():
    input_path = "images/lena.png"
    output_path = "images/lena_copy.png"

    try:
        img = load_image(input_path)

        print_image_info(img, title="Yüklenen Görüntü Bilgisi")

        save_image(img, output_path)
        print(f"Görüntü başarıyla kaydedildi: {output_path}")

    except FileNotFoundError as e:
        print(f"Hata: {e}")
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")


if __name__ == "__main__":
    main()