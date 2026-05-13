import numpy as np
import math


# ──────────────────────────────────────────────
#  RGB → HSI  Dönüşümü
# ──────────────────────────────────────────────
#  H (Hue / Ton)       : 0 – 360°
#  S (Saturation / Doyg): 0 – 1
#  I (Intensity / Yoğ.) : 0 – 1
#
#  Formüller:
#    I = (R + G + B) / 3
#    S = 1 - [ 3 × min(R,G,B) / (R+G+B) ]
#    θ = arccos { 0.5×[(R-G)+(R-B)] / sqrt[(R-G)²+(R-B)(G-B)] }
#    H = θ         eğer B ≤ G
#    H = 360 - θ   eğer B > G
# ──────────────────────────────────────────────

def rgb_to_hsi_manual(img: np.ndarray) -> np.ndarray:
    """
    RGB görüntüyü HSI renk uzayına çevirir.
    Girdi : uint8 RGB (H×W×3)
    Çıktı : float64 HSI (H×W×3)  →  H: 0-360, S: 0-1, I: 0-1
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("RGB (H×W×3) görüntü gerekli.")

    h, w, _ = img.shape
    hsi = np.zeros((h, w, 3), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            R = float(img[i, j, 0]) / 255.0
            G = float(img[i, j, 1]) / 255.0
            B = float(img[i, j, 2]) / 255.0

            # ── Intensity ──
            I = (R + G + B) / 3.0

            # ── Saturation ──
            min_val = min(R, G, B)
            total = R + G + B
            if total == 0:
                S = 0.0
            else:
                S = 1.0 - (3.0 * min_val / total)

            # ── Hue ──
            numerator = 0.5 * ((R - G) + (R - B))
            denominator = math.sqrt((R - G) ** 2 + (R - B) * (G - B))

            if denominator == 0:
                H = 0.0
            else:
                theta = math.acos(max(-1.0, min(1.0, numerator / denominator)))
                theta = math.degrees(theta)

                if B <= G:
                    H = theta
                else:
                    H = 360.0 - theta

            hsi[i, j, 0] = H
            hsi[i, j, 1] = S
            hsi[i, j, 2] = I

    return hsi


# ──────────────────────────────────────────────
#  HSI → RGB  Dönüşümü
# ──────────────────────────────────────────────

def hsi_to_rgb_manual(hsi: np.ndarray) -> np.ndarray:
    """
    HSI görüntüyü RGB renk uzayına geri çevirir.
    Girdi : float64 HSI (H×W×3)  →  H: 0-360, S: 0-1, I: 0-1
    Çıktı : uint8 RGB (H×W×3)
    """
    if len(hsi.shape) != 3 or hsi.shape[2] != 3:
        raise ValueError("HSI (H×W×3) görüntü gerekli.")

    h, w, _ = hsi.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            H = hsi[i, j, 0]
            S = hsi[i, j, 1]
            I = hsi[i, j, 2]

            # H değerini 0-360 aralığına sabitle
            H = H % 360.0

            if S == 0:
                # Gri ton (achromatik)
                R = G = B = I
            elif 0 <= H < 120:
                H_rad = math.radians(H)
                B = I * (1.0 - S)
                R = I * (1.0 + (S * math.cos(H_rad)) / math.cos(math.radians(60.0) - H_rad))
                G = 3.0 * I - (R + B)
            elif 120 <= H < 240:
                H = H - 120.0
                H_rad = math.radians(H)
                R = I * (1.0 - S)
                G = I * (1.0 + (S * math.cos(H_rad)) / math.cos(math.radians(60.0) - H_rad))
                B = 3.0 * I - (R + G)
            else:  # 240 <= H < 360
                H = H - 240.0
                H_rad = math.radians(H)
                G = I * (1.0 - S)
                B = I * (1.0 + (S * math.cos(H_rad)) / math.cos(math.radians(60.0) - H_rad))
                R = 3.0 * I - (G + B)

            # 0-255 aralığına çevir ve sınırla
            R = max(0, min(255, int(round(R * 255.0))))
            G = max(0, min(255, int(round(G * 255.0))))
            B = max(0, min(255, int(round(B * 255.0))))

            rgb[i, j, 0] = R
            rgb[i, j, 1] = G
            rgb[i, j, 2] = B

    return rgb


# ──────────────────────────────────────────────
#  RGB → CIE XYZ  Dönüşümü
# ──────────────────────────────────────────────
#  1) sRGB gamma düzeltmesi → linear RGB
#  2) Linear RGB × 3×3 matris → XYZ (D65 referans)
#
#  Gamma düzeltme:
#    C_linear = C_srgb / 12.92                     (C_srgb ≤ 0.04045)
#    C_linear = ((C_srgb + 0.055) / 1.055)^2.4     (C_srgb > 0.04045)
#
#  Matris (D65):
#    X = 0.4124564 R + 0.3575761 G + 0.1804375 B
#    Y = 0.2126729 R + 0.7151522 G + 0.0721750 B
#    Z = 0.0193339 R + 0.1191920 G + 0.9503041 B
# ──────────────────────────────────────────────

def _srgb_to_linear(c: float) -> float:
    """sRGB gamma düzeltmesi: sRGB → Linear RGB."""
    if c <= 0.04045:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4


def rgb_to_xyz_manual(img: np.ndarray) -> np.ndarray:
    """
    RGB görüntüyü CIE XYZ renk uzayına çevirir.
    Girdi : uint8 RGB (H×W×3)
    Çıktı : float64 XYZ (H×W×3)
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("RGB (H×W×3) görüntü gerekli.")

    h, w, _ = img.shape
    xyz = np.zeros((h, w, 3), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            # sRGB → [0, 1] aralığına
            R_srgb = float(img[i, j, 0]) / 255.0
            G_srgb = float(img[i, j, 1]) / 255.0
            B_srgb = float(img[i, j, 2]) / 255.0

            # Gamma düzeltmesi → linear
            R_lin = _srgb_to_linear(R_srgb)
            G_lin = _srgb_to_linear(G_srgb)
            B_lin = _srgb_to_linear(B_srgb)

            # Dönüşüm matrisi (D65 beyaz noktası)
            X = 0.4124564 * R_lin + 0.3575761 * G_lin + 0.1804375 * B_lin
            Y = 0.2126729 * R_lin + 0.7151522 * G_lin + 0.0721750 * B_lin
            Z = 0.0193339 * R_lin + 0.1191920 * G_lin + 0.9503041 * B_lin

            xyz[i, j, 0] = X
            xyz[i, j, 1] = Y
            xyz[i, j, 2] = Z

    return xyz


# ──────────────────────────────────────────────
#  CIE XYZ → CIE L*u*v*  Dönüşümü
# ──────────────────────────────────────────────
#  D65 beyaz noktası:  Xn = 0.95047, Yn = 1.00000, Zn = 1.08883
#
#  u' = 4X / (X + 15Y + 3Z)
#  v' = 9Y / (X + 15Y + 3Z)
#
#  u'n = 4Xn / (Xn + 15Yn + 3Zn)
#  v'n = 9Yn / (Xn + 15Yn + 3Zn)
#
#  Y/Yn > 0.008856 ise:  L* = 116 × (Y/Yn)^(1/3) − 16
#  Y/Yn ≤ 0.008856 ise:  L* = 903.3 × (Y/Yn)
#
#  u* = 13 × L* × (u' − u'n)
#  v* = 13 × L* × (v' − v'n)
# ──────────────────────────────────────────────

# D65 beyaz noktası sabitleri
_Xn = 0.95047
_Yn = 1.00000
_Zn = 1.08883
_un_prime = 4.0 * _Xn / (_Xn + 15.0 * _Yn + 3.0 * _Zn)
_vn_prime = 9.0 * _Yn / (_Xn + 15.0 * _Yn + 3.0 * _Zn)


def xyz_to_luv_manual(xyz: np.ndarray) -> np.ndarray:
    """
    CIE XYZ görüntüyü CIE L*u*v* renk uzayına çevirir.
    Girdi : float64 XYZ (H×W×3)
    Çıktı : float64 Luv (H×W×3)  →  L*: 0-100, u*: ~-134..220, v*: ~-140..122
    """
    if len(xyz.shape) != 3 or xyz.shape[2] != 3:
        raise ValueError("XYZ (H×W×3) görüntü gerekli.")

    h, w, _ = xyz.shape
    luv = np.zeros((h, w, 3), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            X = xyz[i, j, 0]
            Y = xyz[i, j, 1]
            Z = xyz[i, j, 2]

            # u', v' hesabı
            denom = X + 15.0 * Y + 3.0 * Z
            if denom == 0:
                u_prime = 0.0
                v_prime = 0.0
            else:
                u_prime = 4.0 * X / denom
                v_prime = 9.0 * Y / denom

            # L* hesabı
            yr = Y / _Yn
            if yr > 0.008856:
                L = 116.0 * (yr ** (1.0 / 3.0)) - 16.0
            else:
                L = 903.3 * yr

            # u*, v* hesabı
            u_star = 13.0 * L * (u_prime - _un_prime)
            v_star = 13.0 * L * (v_prime - _vn_prime)

            luv[i, j, 0] = L
            luv[i, j, 1] = u_star
            luv[i, j, 2] = v_star

    return luv


# ──────────────────────────────────────────────
#  CIE L*u*v* → CIE XYZ  Dönüşümü (Ters)
# ──────────────────────────────────────────────

def luv_to_xyz_manual(luv: np.ndarray) -> np.ndarray:
    """
    CIE L*u*v* görüntüyü CIE XYZ renk uzayına geri çevirir.
    Girdi : float64 Luv (H×W×3)
    Çıktı : float64 XYZ (H×W×3)
    """
    if len(luv.shape) != 3 or luv.shape[2] != 3:
        raise ValueError("Luv (H×W×3) görüntü gerekli.")

    h, w, _ = luv.shape
    xyz = np.zeros((h, w, 3), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            L = luv[i, j, 0]
            u_star = luv[i, j, 1]
            v_star = luv[i, j, 2]

            # Y hesabı
            if L > 7.9996:  # 903.3 × 0.008856 ≈ 8
                Y = _Yn * ((L + 16.0) / 116.0) ** 3
            else:
                Y = _Yn * L / 903.3

            # u', v' geri hesaplama
            if L == 0:
                u_prime = _un_prime
                v_prime = _vn_prime
            else:
                u_prime = u_star / (13.0 * L) + _un_prime
                v_prime = v_star / (13.0 * L) + _vn_prime

            # X, Z hesabı
            if v_prime == 0:
                X = 0.0
                Z = 0.0
            else:
                X = Y * (9.0 * u_prime) / (4.0 * v_prime)
                Z = Y * (12.0 - 3.0 * u_prime - 20.0 * v_prime) / (4.0 * v_prime)

            xyz[i, j, 0] = X
            xyz[i, j, 1] = Y
            xyz[i, j, 2] = Z

    return xyz


# ──────────────────────────────────────────────
#  RGB → CIE L*u*v*  (Kısa yol: RGB→XYZ→Luv)
# ──────────────────────────────────────────────

def rgb_to_luv_manual(img: np.ndarray) -> np.ndarray:
    """
    RGB görüntüyü CIE L*u*v* renk uzayına çevirir.
    RGB → XYZ → L*u*v* adımlarını sırayla uygular.
    Girdi : uint8 RGB (H×W×3)
    Çıktı : float64 Luv (H×W×3)
    """
    xyz = rgb_to_xyz_manual(img)
    return xyz_to_luv_manual(xyz)


# ──────────────────────────────────────────────
#  Görüntüleme için normalize etme (float → uint8)
# ──────────────────────────────────────────────

def normalize_for_display(img_float: np.ndarray) -> np.ndarray:
    """
    Float renk uzayı verilerini (HSI, XYZ, Luv vb.)
    UI'da göstermek için 0-255 uint8 aralığına normalize eder.
    Her kanal ayrı ayrı min-max normalizasyonu yapılır.
    """
    if len(img_float.shape) != 3 or img_float.shape[2] != 3:
        raise ValueError("3 kanallı (H×W×3) görüntü gerekli.")

    h, w, c = img_float.shape
    out = np.zeros((h, w, c), dtype=np.uint8)

    for k in range(c):
        channel = img_float[:, :, k]
        c_min = np.min(channel)
        c_max = np.max(channel)

        if c_max == c_min:
            out[:, :, k] = 0
        else:
            # Min-max normalizasyonu: 0-255 aralığına
            normalized = (channel - c_min) / (c_max - c_min) * 255.0
            out[:, :, k] = np.clip(normalized, 0, 255).astype(np.uint8)

    return out
