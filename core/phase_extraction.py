"""
Phase extraction from interferograms using FFT method.
"""

import numpy as np
from typing import Optional, Tuple
from config.settings import FFT_FILTER_SIGMA, DC_MASK_RADIUS


def extract_phase_fft(
    interferogram: np.ndarray,
    mask: Optional[np.ndarray] = None,
    carrier_frequency: Optional[Tuple[int, int]] = None,
    filter_sigma: float = FFT_FILTER_SIGMA
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract phase from interferogram using Fourier transform method.

    Args:
        interferogram: Input interferogram image (2D array)
        mask: Binary mask (optional)
        carrier_frequency: (fx, fy) carrier frequency in pixels (optional, auto-detect if None)
        filter_sigma: Bandwidth of Gaussian bandpass filter

    Returns:
        wrapped_phase: Phase map in range [-π, π]
        fft_spectrum: FFT spectrum for visualization
    """
    # Convert to float
    img = interferogram.astype(np.float64)

    # Apply mask if provided
    if mask is not None:
        img = img * mask

    # 1. Compute FFT
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)

    # 2. Find carrier frequency (highest peak excluding DC)
    if carrier_frequency is None:
        magnitude = np.abs(fft_shifted)
        h, w = magnitude.shape

        # Mask DC component
        center_y, center_x = h // 2, w // 2
        magnitude[center_y-DC_MASK_RADIUS:center_y+DC_MASK_RADIUS,
                  center_x-DC_MASK_RADIUS:center_x+DC_MASK_RADIUS] = 0

        # Find peak
        peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        carrier_frequency = (peak_idx[1] - center_x, peak_idx[0] - center_y)

    # 3. Create bandpass filter centered at carrier frequency
    h, w = interferogram.shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2

    # Gaussian bandpass filter
    fx, fy = carrier_frequency
    bandpass = np.exp(-((x - (center_x + fx))**2 + (y - (center_y + fy))**2) / (2 * filter_sigma**2))

    # 4. Apply filter
    filtered_fft = fft_shifted * bandpass

    # 5. Shift back and inverse FFT
    filtered_fft = np.fft.ifftshift(filtered_fft)
    complex_field = np.fft.ifft2(filtered_fft)

    # 6. Extract phase
    wrapped_phase = np.arctan2(complex_field.imag, complex_field.real)

    # Apply mask to phase
    if mask is not None:
        wrapped_phase = wrapped_phase * mask

    return wrapped_phase, fft_shifted


def get_fft_spectrum(image: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """
    Compute FFT spectrum for visualization.

    Args:
        image: Input image
        log_scale: Whether to apply log scale

    Returns:
        FFT magnitude spectrum
    """
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    if log_scale:
        magnitude = np.log(1 + magnitude)

    # Normalize for visualization
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())

    return magnitude


def find_carrier_frequency(
    interferogram: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[int, int]:
    """
    Automatically find carrier frequency from interferogram.

    Args:
        interferogram: Input interferogram
        mask: Binary mask (optional)

    Returns:
        (fx, fy) carrier frequency coordinates
    """
    img = interferogram.astype(np.float64)

    if mask is not None:
        img = img * mask

    # FFT
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    # Mask DC component
    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2
    magnitude[center_y-DC_MASK_RADIUS:center_y+DC_MASK_RADIUS,
              center_x-DC_MASK_RADIUS:center_x+DC_MASK_RADIUS] = 0

    # Find peak
    peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    fx = peak_idx[1] - center_x
    fy = peak_idx[0] - center_y

    return fx, fy
