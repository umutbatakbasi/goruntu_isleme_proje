from utils.helpers import get_image_info


def print_image_info(img, title: str = "Görüntü Bilgisi") -> None:
    """
    Görüntü bilgisini terminale yazdırır.
    """
    info = get_image_info(img)

    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)

    for key, value in info.items():
        print(f"{key}: {value}")

    print("=" * 40 + "\n")