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

    # 5. Inverse FFT (result still has carrier modulation)
    filtered_fft_unshifted = np.fft.ifftshift(filtered_fft)
    complex_field = np.fft.ifft2(filtered_fft_unshifted)

    # 6. CRITICAL: Remove carrier modulation in spatial domain
    # After filtering around carrier (fx, fy) and IFFT, the complex field is:
    # complex_field[y,x] = amplitude * exp(i*phase) * exp(i*2π*(fx*x/w + fy*y/h))
    # We multiply by conjugate to remove the carrier oscillation
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    # Carrier phase ramp (conjugate has negative sign)
    carrier_phase = -2.0 * np.pi * (fx * xx / w + fy * yy / h)
    carrier_modulation = np.exp(1j * carrier_phase)

    # Demodulate by multiplying with conjugate of carrier
    complex_field_demod = complex_field * carrier_modulation

    # 7. Extract phase from demodulated complex field
    wrapped_phase = np.arctan2(complex_field_demod.imag, complex_field_demod.real)

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
