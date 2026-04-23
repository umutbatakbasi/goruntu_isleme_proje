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

from filters.noise import salt_pepper_noise_manual
from filters.mean_filter import mean_filter_manual
from filters.median_filter import median_filter_manual
from filters.motion_filter import motion_filter_manual

from edge_threshold.double_threshold import double_threshold_manual
from edge_threshold.canny_like import canny_like_manual

from morphology.dilate import dilate_manual
from morphology.erode import erode_manual
from morphology.opening import opening_manual
from morphology.closing import closing_manual

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

        # 14) Salt & Pepper gürültü ekleme
        noisy = salt_pepper_noise_manual(gray, noise_ratio=0.05)
        """
        noisy = salt_pepper_noise_manual(gray, noise_ratio=0.02) Hafif Gürültü
        noisy = salt_pepper_noise_manual(gray, noise_ratio=0.10) Belirgin Gürültü
        """
        print_image_info(noisy, "Salt & Pepper Gürültülü Görüntü")
        save_image(noisy, "images/lena_salt_pepper.png")

        hist_noisy = histogram_manual(noisy)
        print_histogram_summary(hist_noisy)
        plot_histogram(hist_noisy, title="Salt & Pepper Gürültülü Görüntü Histogramı")

        # 15) Mean filter
        mean_filtered = mean_filter_manual(noisy, kernel_size=3)
        print_image_info(mean_filtered, "Mean Filter Uygulanmış Görüntü")
        save_image(mean_filtered, "images/lena_mean_filtered.png")

        hist_mean = histogram_manual(mean_filtered)
        print_histogram_summary(hist_mean)
        plot_histogram(hist_mean, title="Mean Filter Sonrası Histogram")

        # 16) Median filter
        median_filtered = median_filter_manual(noisy, kernel_size=3)
        print_image_info(median_filtered, "Median Filter Uygulanmış Görüntü")
        save_image(median_filtered, "images/lena_median_filtered.png")

        hist_median = histogram_manual(median_filtered)
        print_histogram_summary(hist_median)
        plot_histogram(hist_median, title="Median Filter Sonrası Histogram")

        # 17) Motion filter
        motion_horizontal = motion_filter_manual(gray, kernel_size=9, direction="horizontal")
        print_image_info(motion_horizontal, "Horizontal Motion Filter Uygulanmış Görüntü")
        save_image(motion_horizontal, "images/lena_motion_horizontal.png")

        hist_motion_h = histogram_manual(motion_horizontal)
        print_histogram_summary(hist_motion_h)
        plot_histogram(hist_motion_h, title="Horizontal Motion Filter Sonrası Histogram")

        motion_vertical = motion_filter_manual(gray, kernel_size=9, direction="vertical")
        print_image_info(motion_vertical, "Vertical Motion Filter Uygulanmış Görüntü")
        save_image(motion_vertical, "images/lena_motion_vertical.png")

        # 18) Double threshold
        double_thresh = double_threshold_manual(
            gray,
            low_threshold=70,
            high_threshold=150,
            weak_value=128,
            strong_value=255
        )
        print_image_info(double_thresh, "Çift Eşiklenmiş Görüntü")
        save_image(double_thresh, "images/lena_double_threshold.png")

        hist_double = histogram_manual(double_thresh)
        print_histogram_summary(hist_double)
        plot_histogram(hist_double, title="Çift Eşikleme Sonrası Histogram")

        # 19) Canny-like edge detection
        smoothed_img, gradient_mag, canny_like_edges = canny_like_manual(
            gray,
            smoothing_kernel_size=3,
            low_threshold=40,
            high_threshold=100,
            weak_value=128,
            strong_value=255
        )

        print_image_info(smoothed_img, "Canny-Like: Yumuşatılmış Görüntü")
        save_image(smoothed_img, "images/lena_canny_smoothed.png")

        print_image_info(gradient_mag, "Canny-Like: Gradient Magnitude")
        save_image(gradient_mag, "images/lena_canny_gradient.png")

        print_image_info(canny_like_edges, "Canny-Like: Final Kenar Görüntüsü")
        save_image(canny_like_edges, "images/lena_canny_like_edges.png")

        hist_canny = histogram_manual(canny_like_edges)
        print_histogram_summary(hist_canny)
        plot_histogram(hist_canny, title="Canny-Like Final Kenar Histogramı")

        # 20) Morphology - Dilation
        dilated_square = dilate_manual(binary, se_size=3, se_shape="square")
        print_image_info(dilated_square, "Dilation Uygulanmış Binary Görüntü (Square)")
        save_image(dilated_square, "images/lena_dilated_square.png")

        hist_dilated = histogram_manual(dilated_square)
        print_histogram_summary(hist_dilated)
        plot_histogram(hist_dilated, title="Dilation Sonrası Histogram (Square)")

        dilated_cross = dilate_manual(binary, se_size=3, se_shape="cross")
        print_image_info(dilated_cross, "Dilation Uygulanmış Binary Görüntü (Cross)")
        save_image(dilated_cross, "images/lena_dilated_cross.png")

        # 21) Morphology - Erosion
        eroded_square = erode_manual(binary, se_size=3, se_shape="square")
        print_image_info(eroded_square, "Erosion Uygulanmış Binary Görüntü (Square)")
        save_image(eroded_square, "images/lena_eroded_square.png")

        hist_eroded = histogram_manual(eroded_square)
        print_histogram_summary(hist_eroded)
        plot_histogram(hist_eroded, title="Erosion Sonrası Histogram (Square)")

        eroded_cross = erode_manual(binary, se_size=3, se_shape="cross")
        print_image_info(eroded_cross, "Erosion Uygulanmış Binary Görüntü (Cross)")
        save_image(eroded_cross, "images/lena_eroded_cross.png")

        # 22) Morphology - Opening
        opened_square = opening_manual(binary, se_size=3, se_shape="square")
        print_image_info(opened_square, "Opening Uygulanmış Binary Görüntü (Square)")
        save_image(opened_square, "images/lena_opened_square.png")

        hist_opened = histogram_manual(opened_square)
        print_histogram_summary(hist_opened)
        plot_histogram(hist_opened, title="Opening Sonrası Histogram (Square)")

        opened_cross = opening_manual(binary, se_size=3, se_shape="cross")
        print_image_info(opened_cross, "Opening Uygulanmış Binary Görüntü (Cross)")
        save_image(opened_cross, "images/lena_opened_cross.png")

        # 23) Morphology - Closing
        closed_square = closing_manual(binary, se_size=3, se_shape="square")
        print_image_info(closed_square, "Closing Uygulanmış Binary Görüntü (Square)")
        save_image(closed_square, "images/lena_closed_square.png")

        hist_closed = histogram_manual(closed_square)
        print_histogram_summary(hist_closed)
        plot_histogram(hist_closed, title="Closing Sonrası Histogram (Square)")

        closed_cross = closing_manual(binary, se_size=3, se_shape="cross")
        print_image_info(closed_cross, "Closing Uygulanmış Binary Görüntü (Cross)")
        save_image(closed_cross, "images/lena_closed_cross.png")

        print("Tüm mevcut işlemler başarıyla tamamlandı.")

    except FileNotFoundError as e:
        print(f"Hata: {e}")
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")


if __name__ == "__main__":
    main()