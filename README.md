# Görüntü İşleme Projesi

## Proje Bilgileri

| Bilgi | Açıklama |
|---|---|
| Ders | Görüntü İşleme |
| Üniversite | Selçuk Üniversitesi |
| Fakülte | Teknoloji Fakültesi |
| Bölüm | Bilgisayar Mühendisliği |
| Öğrenci Ad Soyad |  Umutcan Batakbaşı |
| Öğrenci Numarası | 233301108 |
| Öğrenci Ad Soyad |  Ayşe Ceyda Al |
| Öğrenci Numarası | 223301054 |
| Öğrenci Ad Soyad |  Ali Efekan Erkan |
| Öğrenci Numarası | 223301158 |
| Öğretim Üyesi | DR ÖĞRETİM ÜYESİ İLKAY ÇINAR |

---

# Proje Hakkında

Bu proje, Görüntü İşleme dersi kapsamında geliştirilmiş manuel görüntü işleme uygulamasıdır.  
Projede kullanılan algoritmalar mümkün olduğunca matematiksel temelden manuel olarak gerçekleştirilmiştir.

Projede hazır görüntü işleme fonksiyonları kullanılmamış, işlemler NumPy tabanlı matris işlemleri ile gerçekleştirilmiştir.

Amaç:
- Görüntü işleme mantığını temelden öğrenmek
- Piksel seviyesinde işlem yapabilmek
- Filtreleme ve dönüşüm algoritmalarını matematiksel olarak anlayabilmek
- Temel görüntü işleme tekniklerini manuel olarak uygulayabilmektir

---

# Kullanılan Teknolojiler

- Python 3
- NumPy
- Pillow (PIL)
- Matplotlib

---

# Proje Yapısı

```text
goruntu_isleme_proje/
│
├── main.py
├── ui.py
├── requirements.txt
│
├── utils/
│   ├── image_io.py
│   ├── display.py
│   └── helpers.py
│
├── geometric/
│   ├── flip.py
│   ├── rotate.py
│   ├── crop.py
│   └── resize.py
│
├── intensity/
│   ├── grayscale.py
│   ├── binary.py
│   ├── histogram.py
│   ├── contrast.py
│   └── arithmetic.py
│
├── filters/
│   ├── mean_filter.py
│   ├── median_filter.py
│   ├── motion_filter.py
│   └── noise.py
│
├── edge_threshold/
│   ├── double_threshold.py
│   └── canny_like.py
│
├── morphology/
│   ├── dilate.py
│   ├── erode.py
│   ├── opening.py
│   └── closing.py
│
└── images/
    ├── test1.png
    ├── test2.png
    └── lena.png
```
---
Gerçekleştirilen İşlemler
1) Geometrik İşlemler
Horizontal Flip
Görüntünün yatay eksende ters çevrilmesi işlemi gerçekleştirilmiştir.
Vertical Flip
Görüntünün dikey eksende ters çevrilmesi işlemi gerçekleştirilmiştir.
Rotation
Görüntü manuel dönüşüm matrisi kullanılarak belirli açılarda döndürülmüştür.
Crop
Kullanıcının belirlediği koordinatlar arasındaki bölge kırpılmıştır.
Resize
Nearest Neighbor yöntemi ile görüntü boyutlandırma işlemi uygulanmıştır.
---
2) Yoğunluk Dönüşümleri
RGB → Grayscale
RGB görüntü gri seviyeye dönüştürülmüştür.
Binary Thresholding
Belirli threshold değerine göre görüntü siyah-beyaz hale getirilmiştir.
Histogram Hesaplama
Görüntü histogramı manuel olarak hesaplanmıştır.
Histogram Stretching
Kontrast artırımı için histogram genişletme işlemi uygulanmıştır.
Arithmetic Operations
Görüntü toplama, çıkarma ve çarpma işlemleri gerçekleştirilmiştir.
---
3) Filtreleme İşlemleri
Mean Filter
Komşu piksellerin ortalaması alınarak görüntü yumuşatma işlemi uygulanmıştır.
Median Filter
Gürültü azaltma amacıyla medyan filtre uygulanmıştır.
Motion Filter
Hareket bulanıklığı efekti oluşturulmuştur.
Salt & Pepper Noise
Görüntüye rastgele gürültü eklenmiştir.
---
4) Kenar Algılama ve Threshold İşlemleri
Double Threshold
Düşük ve yüksek eşik değerleri kullanılarak kenar ayrımı yapılmıştır.
Canny Benzeri Kenar Algılama
Gradient hesaplama, threshold ve hysteresis mantığı ile manuel kenar algılama sistemi geliştirilmiştir.
---
5) Morfolojik İşlemler
Dilation
Nesne bölgeleri genişletilmiştir.
Erosion
Nesne bölgeleri küçültülmüştür.
Opening
Erosion ardından dilation uygulanarak küçük gürültüler temizlenmiştir.
Closing
Dilation ardından erosion uygulanarak boşluk doldurma işlemi gerçekleştirilmiştir.
---
Matematiksel Temeller
Projede aşağıdaki temel görüntü işleme yaklaşımları kullanılmıştır:
Konvolüsyon (Convolution)
Kernel Tabanlı İşlemler
Ortalama Hesaplama
Medyan Hesaplama
Gradient Hesaplama
Thresholding
Piksel Dönüşümleri
Morfolojik Operasyonlar
Komşuluk Tabanlı İşlemler
---
Renk Uzayı Dönüşümleri
Projede görüntü işleme teorisi kapsamında aşağıdaki renk uzayları incelenmiştir:
RGB
CIE XYZ
CIE Luv*
Dönüşüm mantıkları matematiksel olarak analiz edilmiştir.
---
Programın Çalıştırılması
Gerekli Kütüphanelerin Kurulması
```bash
pip install -r requirements.txt
```
---
Programın Başlatılması
```bash
python main.py
```
veya
```bash
python ui.py
```
---
Proje Amacı
Bu proje sayesinde:
Görüntü işleme algoritmalarının temel mantığı öğrenilmiştir.
Piksel seviyesinde işlem yapma becerisi kazanılmıştır.
Matematiksel görüntü işleme yöntemleri uygulanmıştır.
Manuel algoritma geliştirme deneyimi elde edilmiştir.
---
Teşekkür
Bu proje sürecinde bizlere yol gösteren, görüntü işleme konularını öğrenmemizde destek sağlayan değerli hocamız  
İLKAY ÇIKAR’a teşekkürlerimizi sunarız.
---
Not
Bu proje eğitim amaçlı geliştirilmiştir.

