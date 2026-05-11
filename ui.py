import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np

from utils.image_io import load_image, save_image
from intensity.grayscale import rgb_to_gray_manual
from intensity.binary import gray_to_binary_manual
from intensity.histogram import histogram_manual, plot_histogram
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


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Projesi")
        self.root.geometry("1400x800")

        self.original_image = None
        self.second_image = None
        self.result_image = None

        self.original_photo = None
        self.result_photo = None

        self.build_ui()

    def build_ui(self):
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(fill="x")

        tk.Button(top_frame, text="1. Görüntü Yükle", command=self.load_first_image, width=18).pack(side="left", padx=5)
        tk.Button(top_frame, text="2. Görüntü Yükle", command=self.load_second_image, width=18).pack(side="left", padx=5)
        tk.Button(top_frame, text="Sonucu Kaydet", command=self.save_result, width=18).pack(side="left", padx=5)
        tk.Button(top_frame, text="Histogram Göster", command=self.show_histogram, width=18).pack(side="left", padx=5)

        control_frame = tk.LabelFrame(self.root, text="İşlem Kontrol Paneli", padx=10, pady=10)
        control_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(control_frame, text="İşlem Seç:").grid(row=0, column=0, sticky="w")

        self.operation_var = tk.StringVar()
        self.operation_menu = ttk.Combobox(
            control_frame,
            textvariable=self.operation_var,
            width=35,
            state="readonly",
            values=[
                "RGB -> Gray",
                "Gray -> Binary",
                "Flip Horizontal",
                "Flip Vertical",
                "Crop",
                "Resize",
                "Rotate",
                "Histogram Stretch",
                "Contrast Reduce",
                "Add Images",
                "Subtract Images",
                "Multiply Images",
                "Salt & Pepper Noise",
                "Mean Filter",
                "Median Filter",
                "Motion Filter",
                "Double Threshold",
                "Canny Like",
                "Dilate",
                "Erode",
                "Opening",
                "Closing",
            ],
        )
        self.operation_menu.grid(row=0, column=1, padx=5, pady=5)
        self.operation_menu.current(0)

        tk.Label(control_frame, text="Parametre 1:").grid(row=1, column=0, sticky="w")
        self.param1_entry = tk.Entry(control_frame, width=20)
        self.param1_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        tk.Label(control_frame, text="Parametre 2:").grid(row=1, column=2, sticky="w")
        self.param2_entry = tk.Entry(control_frame, width=20)
        self.param2_entry.grid(row=1, column=3, sticky="w", padx=5, pady=5)

        tk.Label(control_frame, text="Parametre 3:").grid(row=2, column=0, sticky="w")
        self.param3_entry = tk.Entry(control_frame, width=20)
        self.param3_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        tk.Label(control_frame, text="Yapısal Eleman / Yön:").grid(row=2, column=2, sticky="w")
        self.option_entry = tk.Entry(control_frame, width=20)
        self.option_entry.grid(row=2, column=3, sticky="w", padx=5, pady=5)

        tk.Button(control_frame, text="İşlemi Uygula", command=self.apply_operation, width=20).grid(
            row=3, column=0, columnspan=2, pady=10
        )

        tk.Button(control_frame, text="Parametre Yardımı", command=self.show_help, width=20).grid(
            row=3, column=2, columnspan=2, pady=10
        )

        image_frame = tk.Frame(self.root)
        image_frame.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame = tk.LabelFrame(image_frame, text="Orijinal Görüntü", padx=10, pady=10)
        left_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.original_label = tk.Label(left_frame, text="Görüntü yüklenmedi")
        self.original_label.pack(fill="both", expand=True)

        right_frame = tk.LabelFrame(image_frame, text="Sonuç Görüntüsü", padx=10, pady=10)
        right_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.result_label = tk.Label(right_frame, text="Henüz işlem uygulanmadı")
        self.result_label.pack(fill="both", expand=True)

        info_frame = tk.LabelFrame(self.root, text="Bilgi", padx=10, pady=10)
        info_frame.pack(fill="x", padx=10, pady=10)

        self.info_text = tk.Text(info_frame, height=8)
        self.info_text.pack(fill="x")

    def log(self, text):
        self.info_text.insert(tk.END, text + "\n")
        self.info_text.see(tk.END)

    def load_first_image(self):
        path = filedialog.askopenfilename(
            title="Birinci görüntüyü seç",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if not path:
            return

        try:
            self.original_image = load_image(path)
            self.display_image(self.original_image, self.original_label, is_original=True)
            self.log(f"1. görüntü yüklendi: {path}")
        except Exception as e:
            messagebox.showerror("Hata", str(e))

    def load_second_image(self):
        path = filedialog.askopenfilename(
            title="İkinci görüntüyü seç",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if not path:
            return

        try:
            self.second_image = load_image(path)
            self.log(f"2. görüntü yüklendi: {path}")
        except Exception as e:
            messagebox.showerror("Hata", str(e))

    def save_result(self):
        if self.result_image is None:
            messagebox.showwarning("Uyarı", "Kaydedilecek sonuç görüntüsü yok.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if not path:
            return

        try:
            save_image(self.result_image, path)
            self.log(f"Sonuç kaydedildi: {path}")
            messagebox.showinfo("Başarılı", "Sonuç görüntüsü kaydedildi.")
        except Exception as e:
            messagebox.showerror("Hata", str(e))

    def show_histogram(self):
        if self.result_image is None:
            messagebox.showwarning("Uyarı", "Önce bir sonuç üretmelisin.")
            return

        if len(self.result_image.shape) != 2:
            messagebox.showwarning("Uyarı", "Histogram için gri seviyeli sonuç görüntüsü gerekli.")
            return

        try:
            hist = histogram_manual(self.result_image)
            plot_histogram(hist, title="Sonuç Görüntüsü Histogramı")
            self.log("Histogram gösterildi.")
        except Exception as e:
            messagebox.showerror("Hata", str(e))

    def display_image(self, img, target_label, is_original=False):
        display_img = img.copy()

        if len(display_img.shape) == 2:
            pil_img = Image.fromarray(display_img.astype(np.uint8))
        else:
            pil_img = Image.fromarray(display_img.astype(np.uint8))

        pil_img.thumbnail((600, 500))
        photo = ImageTk.PhotoImage(pil_img)

        target_label.config(image=photo, text="")
        target_label.image = photo

        if is_original:
            self.original_photo = photo
        else:
            self.result_photo = photo

    def get_int_param(self, entry_widget, default=None):
        text = entry_widget.get().strip()
        if text == "":
            return default
        return int(text)

    def get_float_param(self, entry_widget, default=None):
        text = entry_widget.get().strip()
        if text == "":
            return default
        return float(text)

    def apply_operation(self):
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce birinci görüntüyü yüklemelisin.")
            return

        op = self.operation_var.get()

        try:
            if op == "RGB -> Gray":
                self.result_image = rgb_to_gray_manual(self.original_image)

            elif op == "Gray -> Binary":
                threshold = self.get_int_param(self.param1_entry, 128)
                gray = self.ensure_grayscale(self.original_image)
                self.result_image = gray_to_binary_manual(gray, threshold)

            elif op == "Flip Horizontal":
                self.result_image = flip_horizontal_manual(self.original_image)

            elif op == "Flip Vertical":
                self.result_image = flip_vertical_manual(self.original_image)

            elif op == "Crop":
                x1 = self.get_int_param(self.param1_entry, 50)
                y1 = self.get_int_param(self.param2_entry, 50)
                x2 = self.get_int_param(self.param3_entry, 300)
                option = self.option_entry.get().strip()
                y2 = int(option) if option else 300
                self.result_image = crop_manual(self.original_image, x1, y1, x2, y2)

            elif op == "Resize":
                scale = self.get_float_param(self.param1_entry, 1.5)
                self.result_image = resize_nn_manual(self.original_image, scale)

            elif op == "Rotate":
                angle = self.get_float_param(self.param1_entry, 30.0)
                self.result_image = rotate_manual(self.original_image, angle)

            elif op == "Histogram Stretch":
                gray = self.ensure_grayscale(self.original_image)
                self.result_image = histogram_stretch_manual(gray)

            elif op == "Contrast Reduce":
                factor = self.get_float_param(self.param1_entry, 0.5)
                gray = self.ensure_grayscale(self.original_image)
                self.result_image = contrast_reduce_manual(gray, factor)

            elif op == "Add Images":
                self.check_second_image()
                self.result_image = image_add_manual(self.original_image, self.second_image)

            elif op == "Subtract Images":
                self.check_second_image()
                self.result_image = image_subtract_manual(self.original_image, self.second_image)

            elif op == "Multiply Images":
                self.check_second_image()
                self.result_image = image_multiply_manual(self.original_image, self.second_image)

            elif op == "Salt & Pepper Noise":
                ratio = self.get_float_param(self.param1_entry, 0.05)
                gray = self.ensure_grayscale(self.original_image)
                self.result_image = salt_pepper_noise_manual(gray, ratio)

            elif op == "Mean Filter":
                k = self.get_int_param(self.param1_entry, 3)
                gray = self.ensure_grayscale(self.original_image)
                self.result_image = mean_filter_manual(gray, k)

            elif op == "Median Filter":
                k = self.get_int_param(self.param1_entry, 3)
                gray = self.ensure_grayscale(self.original_image)
                self.result_image = median_filter_manual(gray, k)

            elif op == "Motion Filter":
                k = self.get_int_param(self.param1_entry, 9)
                direction = self.option_entry.get().strip() or "horizontal"
                gray = self.ensure_grayscale(self.original_image)
                self.result_image = motion_filter_manual(gray, k, direction)

            elif op == "Double Threshold":
                low_t = self.get_int_param(self.param1_entry, 70)
                high_t = self.get_int_param(self.param2_entry, 150)
                gray = self.ensure_grayscale(self.original_image)
                self.result_image = double_threshold_manual(gray, low_t, high_t)

            elif op == "Canny Like":
                low_t = self.get_int_param(self.param1_entry, 40)
                high_t = self.get_int_param(self.param2_entry, 100)
                smooth_k = self.get_int_param(self.param3_entry, 3)
                gray = self.ensure_grayscale(self.original_image)
                _, _, edges = canny_like_manual(
                    gray,
                    smoothing_kernel_size=smooth_k,
                    low_threshold=low_t,
                    high_threshold=high_t,
                )
                self.result_image = edges

            elif op == "Dilate":
                se_size = self.get_int_param(self.param1_entry, 3)
                se_shape = self.option_entry.get().strip() or "square"
                binary = self.ensure_binary(self.original_image)
                self.result_image = dilate_manual(binary, se_size, se_shape)

            elif op == "Erode":
                se_size = self.get_int_param(self.param1_entry, 3)
                se_shape = self.option_entry.get().strip() or "square"
                binary = self.ensure_binary(self.original_image)
                self.result_image = erode_manual(binary, se_size, se_shape)

            elif op == "Opening":
                se_size = self.get_int_param(self.param1_entry, 3)
                se_shape = self.option_entry.get().strip() or "square"
                binary = self.ensure_binary(self.original_image)
                self.result_image = opening_manual(binary, se_size, se_shape)

            elif op == "Closing":
                se_size = self.get_int_param(self.param1_entry, 3)
                se_shape = self.option_entry.get().strip() or "square"
                binary = self.ensure_binary(self.original_image)
                self.result_image = closing_manual(binary, se_size, se_shape)

            else:
                messagebox.showwarning("Uyarı", "Geçerli bir işlem seç.")
                return

            self.display_image(self.result_image, self.result_label, is_original=False)
            self.log(f"İşlem uygulandı: {op}")

        except Exception as e:
            messagebox.showerror("Hata", str(e))
            self.log(f"Hata: {str(e)}")

    def ensure_grayscale(self, img):
        if len(img.shape) == 2:
            return img
        return rgb_to_gray_manual(img)

    def ensure_binary(self, img):
        gray = self.ensure_grayscale(img)
        return gray_to_binary_manual(gray, 128)

    def check_second_image(self):
        if self.second_image is None:
            raise ValueError("Bu işlem için ikinci görüntü yüklemelisin.")

    def show_help(self):
        help_text = """
Parametre Yardımı

RGB -> Gray
- Parametre gerekmez

Gray -> Binary
- Parametre 1: threshold (örnek: 128)

Crop
- Parametre 1: x1
- Parametre 2: y1
- Parametre 3: x2
- Yapısal Eleman / Yön: y2

Resize
- Parametre 1: scale (örnek: 0.5 veya 1.5)

Rotate
- Parametre 1: angle (örnek: 30, -45)

Contrast Reduce
- Parametre 1: factor (0 ile 1 arası)

Salt & Pepper Noise
- Parametre 1: noise ratio (örnek: 0.05)

Mean / Median Filter
- Parametre 1: kernel size (3, 5, 7)

Motion Filter
- Parametre 1: kernel size
- Yapısal Eleman / Yön:
  horizontal / vertical / diag_main / diag_anti

Double Threshold
- Parametre 1: low threshold
- Parametre 2: high threshold

Canny Like
- Parametre 1: low threshold
- Parametre 2: high threshold
- Parametre 3: smoothing kernel size

Dilate / Erode / Opening / Closing
- Parametre 1: structuring element size
- Yapısal Eleman / Yön:
  square / cross
"""
        messagebox.showinfo("Parametre Yardımı", help_text)


def run_ui():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_ui()