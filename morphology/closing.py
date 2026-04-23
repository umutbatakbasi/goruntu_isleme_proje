import numpy as np
from morphology.dilate import dilate_manual
from morphology.erode import erode_manual


def closing_manual(img: np.ndarray, se_size: int = 3, se_shape: str = "square") -> np.ndarray:
    """
    Binary görüntüye closing uygular.
    Kapama = dilation + erosion
    """

    dilated = dilate_manual(img, se_size=se_size, se_shape=se_shape)
    closed = erode_manual(dilated, se_size=se_size, se_shape=se_shape)

    return closed