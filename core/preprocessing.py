"""
Image preprocessing functions for interferogram analysis.
Includes denoising, smoothing, vignette removal, and fringe removal.
"""

import numpy as np
import cv2
from skimage.restoration import denoise_tv_chambolle
from typing import List, Tuple


def denoise_nonlocal_means(
    image: np.ndarray,
    h: float = 10,
    template_window_size: int = 7,
    search_window_size: int = 21
) -> np.ndarray:
    """
    Apply non-local means denoising.

    Args:
        image: Input image (uint8)
        h: Filter strength (higher = more denoising)
        template_window_size: Size of template patch
        search_window_size: Size of search area

    Returns:
        Denoised image
    """
    if image.dtype != np.uint8:
        image = (image * 255 / image.max()).astype(np.uint8)

    denoised = cv2.fastNlMeansDenoising(
        image,
        None,
        h=h,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size
    )

    return denoised


def denoise_bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> np.ndarray:
    """
    Apply bilateral filtering (edge-preserving denoising).

    Args:
        image: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Denoised image
    """
    if image.dtype != np.uint8:
        image = (image * 255 / image.max()).astype(np.uint8)

    denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    return denoised


def denoise_median(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median filtering.

    Args:
        image: Input image
        kernel_size: Size of median filter kernel (must be odd)

    Returns:
        Denoised image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd

    denoised = cv2.medianBlur(image, kernel_size)

    return denoised


def smooth_gaussian(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing.

    Args:
        image: Input image
        sigma: Standard deviation of Gaussian kernel

    Returns:
        Smoothed image
    """
    smoothed = cv2.GaussianBlur(image, (0, 0), sigma)

    return smoothed


def smooth_anisotropic(
    image: np.ndarray,
    weight: float = 0.1,
    iterations: int = 50
) -> np.ndarray:
    """
    Apply anisotropic diffusion (edge-preserving smoothing).

    Args:
        image: Input image
        weight: Denoising weight (higher = more smoothing)
        iterations: Number of iterations

    Returns:
        Smoothed image
    """
    # Normalize to 0-1 for skimage
    if image.dtype == np.uint8:
        img_norm = image.astype(np.float32) / 255.0
    else:
        img_norm = image.astype(np.float32)
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())

    # Apply TV denoising (approximation of anisotropic diffusion)
    smoothed = denoise_tv_chambolle(img_norm, weight=weight, max_num_iter=iterations)

    # Convert back to original dtype
    if image.dtype == np.uint8:
        smoothed = (smoothed * 255).astype(np.uint8)
    else:
        smoothed = smoothed.astype(image.dtype)

    return smoothed


def remove_vignette(image: np.ndarray, polynomial_degree: int = 2) -> np.ndarray:
    """
    Remove vignetting by fitting and dividing by polynomial surface.

    Args:
        image: Input image
        polynomial_degree: Degree of polynomial fit

    Returns:
        Vignette-corrected image
    """
    h, w = image.shape

    # Normalize coordinates to [-1, 1]
    y, x = np.ogrid[:h, :w]
    x = (x - w/2) / (w/2)
    y = (y - h/2) / (h/2)

    # Create polynomial terms
    # For degree 2: 1, x, y, x^2, xy, y^2
    terms = []
    for i in range(polynomial_degree + 1):
        for j in range(polynomial_degree + 1 - i):
            terms.append((x**j) * (y**i))

    # Stack terms into design matrix
    A = np.stack(terms, axis=-1).reshape(-1, len(terms))
    b = image.astype(np.float64).flatten()

    # Least squares fit
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Reconstruct fitted surface
        fitted = A @ coeffs
        fitted = fitted.reshape(h, w)

        # Normalize to prevent divide-by-zero
        fitted = np.maximum(fitted, np.percentile(fitted, 5))

        # Correct vignetting
        corrected = image.astype(np.float64) / fitted
        corrected = (corrected / corrected.max() * 255).astype(np.uint8)

        return corrected

    except np.linalg.LinAlgError:
        # If fit fails, return original image
        print("Warning: Vignette removal failed, returning original image")
        return image


def remove_fringes_fft_notch(
    image: np.ndarray,
    notch_frequencies: List[Tuple[int, int]],
    notch_radius: int = 10
) -> np.ndarray:
    """
    Remove parasitic fringes by notch filtering in Fourier domain.

    Args:
        image: Input image
        notch_frequencies: List of (u, v) frequency coordinates to suppress
        notch_radius: Width of notch filter

    Returns:
        Filtered image
    """
    # Convert to float for FFT
    img_float = image.astype(np.float64)

    # FFT
    fft = np.fft.fft2(img_float)
    fft_shifted = np.fft.fftshift(fft)

    # Create notch filter
    h, w = image.shape
    notch_filter = np.ones((h, w), dtype=np.float64)

    center_y, center_x = h // 2, w // 2

    for u, v in notch_frequencies:
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]

        # Notch at (u, v) and (-u, -v) for symmetry
        mask1 = np.exp(-((x - (center_x + u))**2 + (y - (center_y + v))**2) / (2 * notch_radius**2))
        mask2 = np.exp(-((x - (center_x - u))**2 + (y - (center_y - v))**2) / (2 * notch_radius**2))

        notch_filter *= (1 - mask1) * (1 - mask2)

    # Apply filter
    filtered_fft = fft_shifted * notch_filter

    # Inverse FFT
    filtered = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
    filtered = np.abs(filtered)

    # Convert back to uint8
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)

    return filtered


def highpass_filter(image: np.ndarray, cutoff_frequency: float = 0.1) -> np.ndarray:
    """
    Apply high-pass filter to remove low-frequency background.

    Args:
        image: Input image
        cutoff_frequency: Cutoff frequency (0-1, relative to Nyquist)

    Returns:
        Filtered image
    """
    # Convert to float
    img_float = image.astype(np.float64)

    # FFT
    fft = np.fft.fft2(img_float)
    fft_shifted = np.fft.fftshift(fft)

    # Create high-pass filter
    h, w = image.shape
    center_y, center_x = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Gaussian high-pass filter
    max_distance = np.sqrt(center_x**2 + center_y**2)
    sigma = cutoff_frequency * max_distance

    highpass = 1 - np.exp(-(distance**2) / (2 * sigma**2))

    # Apply filter
    filtered_fft = fft_shifted * highpass

    # Inverse FFT
    filtered = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
    filtered = np.abs(filtered)

    # Normalize
    filtered = ((filtered - filtered.min()) / (filtered.max() - filtered.min()) * 255).astype(np.uint8)

    return filtered


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Input image
        clip_limit: Contrast limit
        tile_size: Size of grid for histogram equalization

    Returns:
        Contrast-enhanced image
    """
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(image)

    return enhanced
