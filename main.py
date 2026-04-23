from utils.image_io import load_image, save_image
from utils.display import print_image_info
from intensity.grayscale import rgb_to_gray_manual
from intensity.binary import gray_to_binary_manual
from intensity.histogram import histogram_manual, print_histogram_summary, plot_histogram
from intensity.contrast import histogram_stretch_manual, contrast_reduce_manual
from intensity.arithmetic import (
    image_add_manual,
    image_subtract_manual,
    image_multiply_manual,
)
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

        # 10) Histogram
        hist_gray = histogram_manual(gray)
        print_histogram_summary(hist_gray)
        plot_histogram(hist_gray, title="Orijinal Gri Görüntü Histogramı")

        # 11) Histogram germe
        stretched = histogram_stretch_manual(gray)
        print_image_info(stretched, "Histogram Gerilmiş Görüntü")
        save_image(stretched, "images/lena_stretched.png")

        hist_stretched = histogram_manual(stretched)
        print_histogram_summary(hist_stretched)
        plot_histogram(hist_stretched, title="Histogram Gerilmiş Görüntü Histogramı")

        # 12) Kontrast azaltma
        reduced = contrast_reduce_manual(gray, factor=0.5)
        print_image_info(reduced, "Kontrastı Azaltılmış Görüntü")
        save_image(reduced, "images/lena_contrast_reduced.png")

        hist_reduced = histogram_manual(reduced)
        print_histogram_summary(hist_reduced)
        plot_histogram(hist_reduced, title="Kontrastı Azaltılmış Görüntü Histogramı")

        # 13) Aritmetik işlemler
        added = image_add_manual(img, flipped_h)
        print_image_info(added, "Toplanmış Görüntü")
        save_image(added, "images/lena_added.png")

        subtracted = image_subtract_manual(img, flipped_h)
        print_image_info(subtracted, "Çıkarılmış Görüntü")
        save_image(subtracted, "images/lena_subtracted.png")

        multiplied = image_multiply_manual(img, flipped_h)
        print_image_info(multiplied, "Çarpılmış Görüntü")
        save_image(multiplied, "images/lena_multiplied.png")

        print("Tüm mevcut işlemler başarıyla tamamlandı.")

    except FileNotFoundError as e:
        print(f"Hata: {e}")
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")


if __name__ == "__main__":
    main()