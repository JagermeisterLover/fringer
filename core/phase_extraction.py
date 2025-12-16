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

    # 2. Find carrier frequency (+1 order sideband peak, excluding DC)
    # The FFT has: DC (center), +1 order (carrier+phase), -1 order (conjugate)
    # We want to find and select the +1 order sideband
    if carrier_frequency is None:
        magnitude = np.abs(fft_shifted)
        h, w = magnitude.shape

        # Mask out DC component (average brightness - not useful for phase)
        center_y, center_x = h // 2, w // 2
        magnitude[center_y-DC_MASK_RADIUS:center_y+DC_MASK_RADIUS,
                  center_x-DC_MASK_RADIUS:center_x+DC_MASK_RADIUS] = 0

        # Find the +1 order carrier peak (highest peak after masking DC)
        peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        carrier_frequency = (peak_idx[1] - center_x, peak_idx[0] - center_y)

    # 3. Create bandpass filter CENTERED AT the carrier frequency
    # This SELECTS the sideband (we want this!), doesn't filter it out
    h, w = interferogram.shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2

    # Gaussian bandpass filter centered at carrier (selects +1 order sideband)
    fx, fy = carrier_frequency
    bandpass = np.exp(-((x - (center_x + fx))**2 + (y - (center_y + fy))**2) / (2 * filter_sigma**2))

    print(f"\nFFT Phase Extraction Debug:")
    print(f"  Carrier frequency (sideband location): fx={fx}, fy={fy}")
    print(f"  Filter sigma: {filter_sigma}")

    # 4. Apply filter to SELECT the sideband
    filtered_fft = fft_shifted * bandpass

    # 5. Shift filtered spectrum to DC (this removes carrier frequency)
    # The peak is at array position (h//2 + fy, w//2 + fx) in shifted space
    # Roll to move it to (h//2, w//2) which is DC in shifted space
    filtered_fft_at_dc = np.roll(filtered_fft, shift=(-fy, -fx), axis=(0, 1))

    # 6. Back to unshifted frequency space (DC now at origin)
    filtered_fft_unshifted = np.fft.ifftshift(filtered_fft_at_dc)

    # 7. Inverse FFT to get complex field at baseband (carrier removed)
    complex_field = np.fft.ifft2(filtered_fft_unshifted)

    # 8. Extract phase using np.angle (more robust than arctan2)
    wrapped_phase = np.angle(complex_field)

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
