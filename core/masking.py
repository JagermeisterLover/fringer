"""
Circular mask generation and operations for interferogram analysis.
"""

import numpy as np
import cv2
from typing import Optional, Tuple


def generate_circular_mask(
    image_shape: Tuple[int, int],
    center: Tuple[float, float],
    outer_radius: float,
    inner_radius: float = 0
) -> np.ndarray:
    """
    Generate a circular annular mask.

    Args:
        image_shape: (height, width) of the image
        center: (x, y) center of the circles
        outer_radius: Outer radius in pixels
        inner_radius: Inner radius in pixels (0 for no central obscuration)

    Returns:
        Binary mask (True for valid pixels, False for masked)
    """
    h, w = image_shape
    cx, cy = center

    # Create coordinate grids
    y, x = np.ogrid[:h, :w]

    # Calculate distance from center
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Create mask: True for pixels between inner and outer radius
    mask = (distance >= inner_radius) & (distance <= outer_radius)

    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray, fill_value: float = 0) -> np.ndarray:
    """
    Apply a mask to an image.

    Args:
        image: Input image
        mask: Binary mask (True for valid pixels)
        fill_value: Value to use for masked pixels

    Returns:
        Masked image
    """
    masked_image = image.copy()
    masked_image[~mask] = fill_value
    return masked_image


def auto_detect_circle(image: np.ndarray, detect_inner: bool = True) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """
    Automatically detect circular features in an image using Hough Circle Transform.

    Args:
        image: Input grayscale image
        detect_inner: Whether to detect inner circle (central obscuration)

    Returns:
        Tuple of ((outer_cx, outer_cy, outer_r), (inner_cx, inner_cy, inner_r))
        Returns None for circles that couldn't be detected
    """
    try:
        # Ensure image is uint8
        if image.dtype != np.uint8:
            img_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            img_normalized = image.copy()

        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(img_normalized, (9, 9), 2)

        # Detect outer circle (largest)
        h, w = image.shape
        min_radius = min(h, w) // 10
        max_radius = min(h, w) // 2

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min(h, w) // 2,  # Only detect one main circle
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        outer_params = None
        inner_params = None

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)

            # Get the largest circle (outer)
            if len(circles) > 0:
                largest_circle = max(circles, key=lambda c: c[2])
                outer_params = (float(largest_circle[0]), float(largest_circle[1]), float(largest_circle[2]))

            # Try to detect inner circle (central obscuration)
            if detect_inner and outer_params is not None:
                # Create mask for outer circle region
                mask = generate_circular_mask(image.shape, (outer_params[0], outer_params[1]), outer_params[2])
                masked_image = apply_mask(img_normalized, mask, fill_value=0)

                # Look for dark central region
                inner_circles = cv2.HoughCircles(
                    masked_image,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=min(h, w) // 4,
                    param1=50,
                    param2=20,
                    minRadius=min_radius // 5,
                    maxRadius=int(outer_params[2] * 0.4)  # Inner circle should be smaller
                )

                if inner_circles is not None:
                    inner_circles = np.round(inner_circles[0, :]).astype(int)
                    if len(inner_circles) > 0:
                        # Find circle closest to center
                        cx, cy = outer_params[0], outer_params[1]
                        closest = min(inner_circles, key=lambda c: np.sqrt((c[0]-cx)**2 + (c[1]-cy)**2))
                        inner_params = (float(closest[0]), float(closest[1]), float(closest[2]))

        return outer_params, inner_params

    except Exception as e:
        print(f"Error in auto_detect_circle: {str(e)}")
        return None, None


def create_mask_overlay(
    image_shape: Tuple[int, int],
    mask: np.ndarray,
    alpha: float = 0.3,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Create a colored overlay for visualizing a mask.

    Args:
        image_shape: Shape of the original image
        mask: Binary mask
        alpha: Transparency (0-1)
        color: RGB color for masked regions

    Returns:
        RGBA overlay image
    """
    h, w = image_shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    # Set color for masked regions
    overlay[~mask] = [color[0], color[1], color[2], int(255 * alpha)]

    return overlay


def get_mask_statistics(mask: np.ndarray) -> dict:
    """
    Calculate statistics about a mask.

    Args:
        mask: Binary mask

    Returns:
        Dictionary with mask statistics
    """
    total_pixels = mask.size
    valid_pixels = np.sum(mask)
    masked_pixels = total_pixels - valid_pixels

    stats = {
        'total_pixels': int(total_pixels),
        'valid_pixels': int(valid_pixels),
        'masked_pixels': int(masked_pixels),
        'valid_percentage': float(valid_pixels / total_pixels * 100),
        'masked_percentage': float(masked_pixels / total_pixels * 100),
    }

    return stats
