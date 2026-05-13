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
from intensity.color_convert import (
    rgb_to_hsi_manual,
    hsi_to_rgb_manual,
    rgb_to_xyz_manual,
    rgb_to_luv_manual,
    xyz_to_luv_manual,
    luv_to_xyz_manual,
    normalize_for_display,
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

# ── Renk Paleti ──
BG = "#1e1e2e"
SURFACE = "#2d2d44"
SURFACE2 = "#383854"
ACCENT = "#7c3aed"
ACCENT_HOVER = "#9333ea"
TEXT = "#e2e8f0"
TEXT_DIM = "#94a3b8"
BORDER = "#4a4a6a"
SUCCESS = "#22c55e"
ERROR = "#ef4444"

CATEGORIES = {
    "🎨 Renk Dönüşümleri": [
        "RGB -> Gray", "Gray -> Binary",
        "RGB -> HSI", "HSI -> RGB",
        "RGB -> CIE XYZ", "RGB -> CIE Luv",
        "CIE XYZ -> CIE Luv", "CIE Luv -> CIE XYZ",
    ],
    "📐 Geometrik": [
        "Flip Horizontal", "Flip Vertical",
        "Crop", "Resize", "Rotate",
    ],
    "💡 Yoğunluk": [
        "Histogram Stretch", "Contrast Reduce",
        "Add Images", "Subtract Images", "Multiply Images",
    ],
    "🔧 Filtreler": [
        "Salt & Pepper Noise", "Mean Filter",
        "Median Filter", "Motion Filter",
    ],
    "🔍 Kenar / Eşik": [
        "Double Threshold", "Canny Like",
    ],
    "🔲 Morfoloji": [
        "Dilate", "Erode", "Opening", "Closing",
    ],
}


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Projesi")
        self.root.geometry("1500x900")
        self.root.configure(bg=BG)
        self.root.minsize(1200, 700)

        self.original_image = None
        self.second_image = None
        self.result_image = None
        self.last_float_result = None  # HSI/XYZ/Luv float sonuçları için

        self.original_photo = None
        self.result_photo = None

        self._setup_styles()
        self.build_ui()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=BG, foreground=TEXT, fieldbackground=SURFACE)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TLabelframe", background=BG, foreground=ACCENT, font=("Segoe UI", 11, "bold"))
        style.configure("TLabelframe.Label", background=BG, foreground=ACCENT)
        style.configure("TCombobox", fieldbackground=SURFACE, background=SURFACE2,
                         foreground=TEXT, selectbackground=ACCENT, font=("Segoe UI", 10))
        style.configure("TEntry", fieldbackground=SURFACE, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Accent.TButton", background=ACCENT, foreground="white",
                         font=("Segoe UI", 10, "bold"), padding=(12, 6))
        style.map("Accent.TButton",
                  background=[("active", ACCENT_HOVER), ("pressed", "#6d28d9")])
        style.configure("TButton", background=SURFACE2, foreground=TEXT,
                         font=("Segoe UI", 10), padding=(10, 5))
        style.map("TButton",
                  background=[("active", BORDER)])
        style.configure("TFrame", background=BG)

    def build_ui(self):
        # ── Üst Buton Çubuğu ──
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill="x", padx=15, pady=(12, 6))

        ttk.Button(top_frame, text="📂 1. Görüntü Yükle", command=self.load_first_image,
                   style="TButton").pack(side="left", padx=4)
        ttk.Button(top_frame, text="📂 2. Görüntü Yükle", command=self.load_second_image,
                   style="TButton").pack(side="left", padx=4)
        ttk.Button(top_frame, text="💾 Sonucu Kaydet", command=self.save_result,
                   style="TButton").pack(side="left", padx=4)
        ttk.Button(top_frame, text="📊 Histogram Göster", command=self.show_histogram,
                   style="TButton").pack(side="left", padx=4)

        # ── Kontrol Paneli ──
        control_frame = ttk.LabelFrame(self.root, text="  İşlem Kontrol Paneli  ", padding=12)
        control_frame.pack(fill="x", padx=15, pady=6)

        # Kategori
        ttk.Label(control_frame, text="Kategori:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.category_var = tk.StringVar()
        self.category_menu = ttk.Combobox(control_frame, textvariable=self.category_var,
                                           width=22, state="readonly",
                                           values=list(CATEGORIES.keys()))
        self.category_menu.grid(row=0, column=1, padx=4, pady=4)
        self.category_menu.current(0)
        self.category_menu.bind("<<ComboboxSelected>>", self._on_category_change)

        # İşlem
        ttk.Label(control_frame, text="İşlem:").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        self.operation_var = tk.StringVar()
        self.operation_menu = ttk.Combobox(control_frame, textvariable=self.operation_var,
                                            width=28, state="readonly")
        self.operation_menu.grid(row=0, column=3, padx=4, pady=4)
        self._on_category_change()

        # Parametreler
        ttk.Label(control_frame, text="Parametre 1:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.param1_entry = ttk.Entry(control_frame, width=18)
        self.param1_entry.grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(control_frame, text="Parametre 2:").grid(row=1, column=2, sticky="w", padx=4, pady=4)
        self.param2_entry = ttk.Entry(control_frame, width=18)
        self.param2_entry.grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(control_frame, text="Parametre 3:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        self.param3_entry = ttk.Entry(control_frame, width=18)
        self.param3_entry.grid(row=2, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(control_frame, text="Yapısal Eleman / Yön:").grid(row=2, column=2, sticky="w", padx=4, pady=4)
        self.option_entry = ttk.Entry(control_frame, width=18)
        self.option_entry.grid(row=2, column=3, sticky="w", padx=4, pady=4)

        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=3, column=0, columnspan=4, pady=10)

        ttk.Button(btn_frame, text="▶  İşlemi Uygula", command=self.apply_operation,
                   style="Accent.TButton").pack(side="left", padx=8)
        ttk.Button(btn_frame, text="❓ Parametre Yardımı", command=self.show_help,
                   style="TButton").pack(side="left", padx=8)

        # ── Görüntü Panelleri ──
        image_frame = ttk.Frame(self.root)
        image_frame.pack(fill="both", expand=True, padx=15, pady=6)

        left_frame = ttk.LabelFrame(image_frame, text="  Orijinal Görüntü  ", padding=8)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.original_label = tk.Label(left_frame, text="Görüntü yüklenmedi",
                                        bg=SURFACE, fg=TEXT_DIM, font=("Segoe UI", 11))
        self.original_label.pack(fill="both", expand=True)

        right_frame = ttk.LabelFrame(image_frame, text="  Sonuç Görüntüsü  ", padding=8)
        right_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))
        self.result_label = tk.Label(right_frame, text="Henüz işlem uygulanmadı",
                                      bg=SURFACE, fg=TEXT_DIM, font=("Segoe UI", 11))
        self.result_label.pack(fill="both", expand=True)

        # ── Bilgi Paneli ──
        info_frame = ttk.LabelFrame(self.root, text="  Bilgi  ", padding=6)
        info_frame.pack(fill="x", padx=15, pady=(6, 12))

        scroll = tk.Scrollbar(info_frame)
        scroll.pack(side="right", fill="y")

        self.info_text = tk.Text(info_frame, height=6, bg=SURFACE, fg=TEXT,
                                  insertbackground=TEXT, font=("Consolas", 9),
                                  relief="flat", yscrollcommand=scroll.set)
        self.info_text.pack(fill="x")
        scroll.config(command=self.info_text.yview)

    def _on_category_change(self, event=None):
        cat = self.category_var.get()
        ops = CATEGORIES.get(cat, [])
        self.operation_menu["values"] = ops
        if ops:
            self.operation_menu.current(0)

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
            self.log(f"✅ 1. görüntü yüklendi: {path}")
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
            self.log(f"✅ 2. görüntü yüklendi: {path}")
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
            self.log(f"💾 Sonuç kaydedildi: {path}")
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
            self.log("📊 Histogram gösterildi.")
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
        saved_float = self.last_float_result
        self.last_float_result = None

        try:
            if op == "RGB -> Gray":
                self.result_image = rgb_to_gray_manual(self.original_image)

            elif op == "Gray -> Binary":
                threshold = self.get_int_param(self.param1_entry, 128)
                gray = self.ensure_grayscale(self.original_image)
                self.result_image = gray_to_binary_manual(gray, threshold)

            elif op == "RGB -> HSI":
                hsi = rgb_to_hsi_manual(self.original_image)
                self.last_float_result = hsi
                self.result_image = normalize_for_display(hsi)

            elif op == "HSI -> RGB":
                if saved_float is not None:
                    self.result_image = hsi_to_rgb_manual(saved_float)
                else:
                    messagebox.showinfo("Bilgi", "Önce 'RGB -> HSI' uygulayın, ardından bu işlemi seçin.")
                    return

            elif op == "RGB -> CIE XYZ":
                xyz = rgb_to_xyz_manual(self.original_image)
                self.last_float_result = xyz
                self.result_image = normalize_for_display(xyz)

            elif op == "RGB -> CIE Luv":
                luv = rgb_to_luv_manual(self.original_image)
                self.last_float_result = luv
                self.result_image = normalize_for_display(luv)

            elif op == "CIE XYZ -> CIE Luv":
                if self.last_float_result is not None and self.last_float_result.shape[2] == 3:
                    xyz = self.last_float_result
                else:
                    xyz = rgb_to_xyz_manual(self.original_image)
                luv = xyz_to_luv_manual(xyz)
                self.last_float_result = luv
                self.result_image = normalize_for_display(luv)

            elif op == "CIE Luv -> CIE XYZ":
                if self.last_float_result is not None and self.last_float_result.shape[2] == 3:
                    luv = self.last_float_result
                else:
                    luv = rgb_to_luv_manual(self.original_image)
                xyz = luv_to_xyz_manual(luv)
                self.last_float_result = xyz
                self.result_image = normalize_for_display(xyz)

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
                    gray, smoothing_kernel_size=smooth_k,
                    low_threshold=low_t, high_threshold=high_t,
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
            self.log(f"✅ İşlem uygulandı: {op}")

        except Exception as e:
            messagebox.showerror("Hata", str(e))
            self.log(f"❌ Hata: {str(e)}")

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

── Renk Dönüşümleri ──
RGB -> Gray         : Parametre gerekmez
Gray -> Binary      : P1 = threshold (örn: 128)
RGB -> HSI          : Parametre gerekmez
HSI -> RGB          : Önce RGB->HSI uygulayın
RGB -> CIE XYZ     : Parametre gerekmez
RGB -> CIE Luv     : Parametre gerekmez
CIE XYZ -> CIE Luv : Önce RGB->XYZ uygulayın
CIE Luv -> CIE XYZ : Önce RGB->Luv uygulayın

── Geometrik ──
Crop    : P1=x1, P2=y1, P3=x2, Yön=y2
Resize  : P1 = scale (örn: 0.5, 1.5)
Rotate  : P1 = angle (örn: 30, -45)

── Yoğunluk ──
Contrast Reduce : P1 = factor (0-1 arası)

── Filtreler ──
Salt & Pepper   : P1 = noise ratio (örn: 0.05)
Mean/Median     : P1 = kernel size (3, 5, 7)
Motion Filter   : P1 = kernel size, Yön = horizontal/vertical/diag_main/diag_anti

── Kenar / Eşik ──
Double Threshold : P1 = low, P2 = high
Canny Like       : P1 = low, P2 = high, P3 = smooth kernel

── Morfoloji ──
Dilate/Erode/Opening/Closing : P1 = SE size, Yön = square/cross
"""
        messagebox.showinfo("Parametre Yardımı", help_text)


def run_ui():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_ui()