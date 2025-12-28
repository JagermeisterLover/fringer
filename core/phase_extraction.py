"""
Phase extraction from interferograms using Takeda (Fourier) method.
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Optional, Tuple
from config.settings import FFT_FILTER_SIGMA, DC_MASK_RADIUS


def extract_phase_fft(
    interferogram: np.ndarray,
    mask: Optional[np.ndarray] = None,
    carrier_frequency: Optional[Tuple[int, int]] = None,
    filter_sigma: float = FFT_FILTER_SIGMA
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract phase from interferogram using Takeda (Fourier) method.

    This is the EXACT implementation from the working Tkinter reference code.

    Args:
        interferogram: Input interferogram image (2D array)
        mask: Binary mask (optional)
        carrier_frequency: (fx, fy) carrier frequency in pixels (optional, auto-detect if None)
        filter_sigma: Bandwidth of Gaussian bandpass filter

    Returns:
        wrapped_phase: Phase map in range [-π, π]
        fft_spectrum: FFT spectrum for visualization
    """
    img = interferogram.astype(np.float64)

    # Normalize image
    img = (img - img.mean()) / (img.std() + 1e-12)

    # Apply mask to ignore outside
    if mask is not None:
        img_masked = img * mask
    else:
        img_masked = img

    # Takeda (Fourier) method: 2D FFT, isolate one sideband, inverse FFT -> complex field
    F = fftshift(fft2(img_masked))
    mag = np.abs(F)

    # suppress central low-frequency area when searching for sideband peak
    h, w = img.shape
    cy = h//2
    cx = w//2
    rr = int(min(h, w) * 0.08)  # radius for DC suppression
    yy, xx = np.mgrid[:h, :w]
    dc_mask = (xx - cx)**2 + (yy - cy)**2 <= rr**2
    mag_masked = mag.copy()
    mag_masked[dc_mask] = 0.0

    # Find peak position (one sideband)
    peak_idx = np.unravel_index(np.argmax(mag_masked), mag_masked.shape)
    pk_y, pk_x = peak_idx

    # Create a gaussian filter around that peak in the frequency domain
    sigma = max(8, int(min(h, w) * 0.04))
    gauss = np.exp(-(((yy - pk_y)**2 + (xx - pk_x)**2) / (2.0*sigma*sigma)))

    # Band-pass by multiplying with gaussian
    F_filtered = F * gauss

    # Bring back and inverse FFT
    F_ishift = ifftshift(F_filtered)
    complex_field = ifft2(F_ishift)

    phase_wrapped = np.angle(complex_field)

    print(f"\nTakeda Phase Extraction Debug:")
    print(f"  Sideband peak position: ({pk_x}, {pk_y})")
    print(f"  Filter sigma: {sigma} pixels")

    return phase_wrapped, F


def get_fft_spectrum(image: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """
    Compute FFT spectrum for visualization.

    Args:
        image: Input image
        log_scale: Whether to apply log scale

    Returns:
        FFT magnitude spectrum
    """
    fft_result = fftshift(fft2(image))
    magnitude = np.abs(fft_result)

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

    # FFT using scipy
    fft_result = fftshift(fft2(img))
    magnitude = np.abs(fft_result)

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
