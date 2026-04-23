import numpy as np
from morphology.erode import erode_manual
from morphology.dilate import dilate_manual


def opening_manual(img: np.ndarray, se_size: int = 3, se_shape: str = "square") -> np.ndarray:
    """
    Binary görüntüye opening uygular.
    Açma = erosion + dilation
    """

    eroded = erode_manual(img, se_size=se_size, se_shape=se_shape)
    opened = dilate_manual(eroded, se_size=se_size, se_shape=se_shape)

    return opened