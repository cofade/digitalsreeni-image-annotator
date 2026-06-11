"""Pure image / array helpers extracted from `ImageAnnotator`.

These are deliberately free of any Qt main-window dependency so they can
be unit-tested in isolation and reused by controllers added in later
refactor phases.
"""

import numpy as np
from PyQt6.QtGui import QImage


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj


def normalize_array(array):
    array_float = array.astype(np.float32)

    if array.dtype == np.uint16:
        array_normalized = (array_float - array.min()) / (array.max() - array.min())
    elif array.dtype == np.uint8:
        p_low, p_high = np.percentile(array_float, (0, 100))
        array_normalized = np.clip(array_float, p_low, p_high)
        array_normalized = (array_normalized - p_low) / (p_high - p_low)
    else:
        array_normalized = (array_float - array.min()) / (array.max() - array.min())

    gamma = 1.0
    array_normalized = np.power(array_normalized, gamma)

    return (array_normalized * 255).astype(np.uint8)


def adjust_contrast(image, low_percentile=1, high_percentile=99):
    if image.dtype != np.uint8:
        p_low, p_high = np.percentile(image, (low_percentile, high_percentile))
        image_adjusted = np.clip(image, p_low, p_high)
        image_adjusted = (image_adjusted - p_low) / (p_high - p_low)
        return (image_adjusted * 255).astype(np.uint8)
    return image


def convert_to_8bit_rgb(image_array):
    if image_array.ndim == 2:
        image_8bit = normalize_array(image_array)
        return np.stack((image_8bit,) * 3, axis=-1)
    if image_array.ndim == 3:
        if image_array.shape[2] == 3:
            return normalize_array(image_array)
        if image_array.shape[2] > 3:
            rgb_array = image_array[:, :, :3]
            return normalize_array(rgb_array)
    raise ValueError(f"Unsupported image shape: {image_array.shape}")


def array_to_qimage(array):
    if array.ndim == 2:
        height, width = array.shape
        return QImage(array.data, width, height, width, QImage.Format.Format_Grayscale8)
    if array.ndim == 3 and array.shape[2] == 3:
        height, width, _ = array.shape
        bytes_per_line = 3 * width
        return QImage(array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    raise ValueError(f"Unsupported array shape {array.shape} for conversion to QImage")
