from utils.image_io import load_image, save_image
from utils.display import print_image_info
from intensity.grayscale import rgb_to_gray_manual
from intensity.binary import gray_to_binary_manual
from geometric.flip import flip_horizontal_manual, flip_vertical_manual
from geometric.crop import crop_manual
from geometric.resize import resize_nn_manual
from geometric.rotate import rotate_manual


def main():
    input_path = "images/lena.png"

    try:
        # 1) Orijinal görüntü
        img = load_image(input_path)
        print_image_info(img, "Orijinal Görüntü")

        # 2) Gri dönüşüm
        gray = rgb_to_gray_manual(img)
        print_image_info(gray, "Gri Görüntü")
        save_image(gray, "images/lena_gray.png")

        # 3) Binary dönüşüm
        threshold_value = 128
        binary = gray_to_binary_manual(gray, threshold=threshold_value)
        print_image_info(binary, f"Binary Görüntü (Threshold={threshold_value})")
        save_image(binary, "images/lena_binary.png")

        # 4) Yatay çevirme
        flipped_h = flip_horizontal_manual(img)
        print_image_info(flipped_h, "Yatay Aynalanmış Görüntü")
        save_image(flipped_h, "images/lena_flip_horizontal.png")

        # 5) Dikey çevirme
        flipped_v = flip_vertical_manual(img)
        print_image_info(flipped_v, "Dikey Aynalanmış Görüntü")
        save_image(flipped_v, "images/lena_flip_vertical.png")

        # 6) Kırpma
        cropped = crop_manual(img, x1=150, y1=150, x2=400, y2=400)
        print_image_info(cropped, "Kırpılmış Görüntü")
        save_image(cropped, "images/lena_cropped.png")

        # 7) Yakınlaştırma
        zoom_in = resize_nn_manual(img, scale=1.5)
        print_image_info(zoom_in, "Yakınlaştırılmış Görüntü (1.5x)")
        save_image(zoom_in, "images/lena_zoom_in.png")

        # 8) Uzaklaştırma
        zoom_out = resize_nn_manual(img, scale=0.5)
        print_image_info(zoom_out, "Uzaklaştırılmış Görüntü (0.5x)")
        save_image(zoom_out, "images/lena_zoom_out.png")

        # 9) Döndürme
        rotated_30 = rotate_manual(img, 30)
        print_image_info(rotated_30, "30 Derece Döndürülmüş Görüntü")
        save_image(rotated_30, "images/lena_rotated_30.png")

        rotated_90 = rotate_manual(img, 90)
        print_image_info(rotated_90, "90 Derece Döndürülmüş Görüntü")
        save_image(rotated_90, "images/lena_rotated_90.png")

        print("Gri, binary, flip, crop, resize ve rotate işlemleri tamamlandı.")

    except FileNotFoundError as e:
        print(f"Hata: {e}")
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")


if __name__ == "__main__":
    main()