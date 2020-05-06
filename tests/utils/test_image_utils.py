import numpy as np
from src.utils import image_utils


def test_generate_dummy_alpha_channel():
    # Given an image
    img_bgr = np.zeros((10, 10, 3), np.uint8)
    # when we generate the dummy alpha channel
    img_bgra = image_utils.generate_dummy_alpha_channel(img_bgr)
    # Then the fourth channel is defined with full opacity
    assert isinstance(img_bgr, np.ndarray) and isinstance(img_bgra, np.ndarray)
    assert img_bgr.shape[2] == 3
    assert img_bgra.shape[2] == 4
    assert (img_bgra[:, :, 3] == 255).all()


def test_calculate_googly_center_and_size():
    # Given two eye corners as coordinates and such settings
    eye = [(2, 2), (4, 4)]
    settings = {'size_multiplier': 2, 'random_max_percent_inc': 0}
    # When we calculate the googly eye metrics
    center, size = image_utils.calculate_googly_center_and_size(eye, settings)
    # Then we get the correct values
    assert center == (3, 3)
    assert size == 6
