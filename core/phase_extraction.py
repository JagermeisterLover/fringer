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

    # 1. Compute FFT (don't shift yet)
    h, w = interferogram.shape
    fft = np.fft.fft2(img)

    # 2. Find carrier frequency by analyzing shifted spectrum
    if carrier_frequency is None:
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)

        # Mask out DC component
        center_y, center_x = h // 2, w // 2
        magnitude[center_y-DC_MASK_RADIUS:center_y+DC_MASK_RADIUS,
                  center_x-DC_MASK_RADIUS:center_x+DC_MASK_RADIUS] = 0

        # Find carrier peak in shifted space
        peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        carrier_frequency = (peak_idx[1] - center_x, peak_idx[0] - center_y)
        fx, fy = carrier_frequency
    else:
        fx, fy = carrier_frequency
        fft_shifted = np.fft.fftshift(fft)

    print(f"\nFFT Phase Extraction Debug:")
    print(f"  Carrier frequency: fx={fx}, fy={fy}")
    print(f"  Filter sigma: {filter_sigma}")

    # 3. Create bandpass filter in shifted space
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    bandpass = np.exp(-((x - (center_x + fx))**2 + (y - (center_y + fy))**2) / (2 * filter_sigma**2))

    # 4. Apply filter in shifted space
    filtered_fft_shifted = fft_shifted * bandpass

    # 5. Go back to unshifted space (don't shift to DC yet)
    filtered_fft = np.fft.ifftshift(filtered_fft_shifted)

    # 6. IFFT to get spatial domain (still has carrier modulation)
    complex_field = np.fft.ifft2(filtered_fft)

    # 7. Remove carrier by multiplying with conjugate in spatial domain
    # This is the heterodyne demodulation step
    yy, xx = np.ogrid[:h, :w]
    # Carrier oscillation is exp(2πi*(fx*x/w + fy*y/h))
    # Multiply by conjugate to remove it
    carrier_removal = np.exp(-2j * np.pi * (fx * xx / w + fy * yy / h))
    complex_field_demod = complex_field * carrier_removal

    # 8. Extract phase
    wrapped_phase = np.angle(complex_field_demod)

    # Apply mask to phase - use NaN for invalid regions instead of 0
    # This prevents artificial discontinuities that break unwrapping
    if mask is not None:
        wrapped_phase = np.where(mask.astype(bool), wrapped_phase, np.nan)

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
