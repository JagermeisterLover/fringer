"""
Image loading and validation for interferogram analysis.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple


def load_image(filepath: str) -> Optional[np.ndarray]:
    """
    Load an image file and convert to grayscale if needed.

    Args:
        filepath: Path to the image file

    Returns:
        Grayscale image as numpy array (uint8), or None if loading fails
    """
    try:
        # Check if file exists
        path = Path(filepath)
        if not path.exists():
            print(f"Error: File does not exist: {filepath}")
            return None

        # Load image
        image = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Error: Failed to load image: {filepath}")
            return None

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:  # BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:  # BGRA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        # Convert to uint8 if needed
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        elif image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)

        return image

    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None


def validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that an image is suitable for interferogram analysis.

    Args:
        image: Input image array

    Returns:
        Tuple of (is_valid, message)
    """
    if image is None:
        return False, "Image is None"

    if not isinstance(image, np.ndarray):
        return False, "Image must be a numpy array"

    if len(image.shape) != 2:
        return False, "Image must be 2D (grayscale)"

    h, w = image.shape
    if h < 64 or w < 64:
        return False, "Image is too small (minimum 64x64)"

    if h > 8192 or w > 8192:
        return False, "Image is too large (maximum 8192x8192)"

    if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
        return False, f"Unsupported image dtype: {image.dtype}"

    return True, "Valid"


def get_image_info(image: np.ndarray) -> dict:
    """
    Get information about an image.

    Args:
        image: Input image array

    Returns:
        Dictionary with image information
    """
    if image is None:
        return {}

    info = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': float(np.min(image)),
        'max': float(np.max(image)),
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
    }

    return info


def normalize_image(image: np.ndarray, target_type: type = np.uint8) -> np.ndarray:
    """
    Normalize image to target data type.

    Args:
        image: Input image
        target_type: Target data type (np.uint8, np.float32, etc.)

    Returns:
        Normalized image
    """
    if target_type == np.uint8:
        if image.dtype == np.uint8:
            return image
        else:
            # Normalize to 0-255
            img_min = np.min(image)
            img_max = np.max(image)
            if img_max > img_min:
                normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(image, dtype=np.uint8)
            return normalized

    elif target_type == np.float32 or target_type == np.float64:
        # Normalize to 0-1
        img_min = np.min(image)
        img_max = np.max(image)
        if img_max > img_min:
            normalized = ((image - img_min) / (img_max - img_min)).astype(target_type)
        else:
            normalized = np.zeros_like(image, dtype=target_type)
        return normalized

    else:
        return image.astype(target_type)
