"""
PSF (Point Spread Function) calculation from wavefront data.
"""

import numpy as np
from typing import Tuple, Optional
from config.settings import PSF_FFT_SIZE


def calculate_psf(
    wavefront: np.ndarray,
    wavelength: float = 632.8e-9,
    pupil_diameter: float = 1.0,
    pixel_size: Optional[float] = None,
    fft_size: int = PSF_FFT_SIZE,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """
    Calculate Point Spread Function from wavefront.

    Args:
        wavefront: Wavefront map in meters
        wavelength: Wavelength in meters
        pupil_diameter: Pupil diameter in meters
        pixel_size: Pixel size in pupil plane (meters)
        fft_size: FFT size for oversampling
        mask: Binary mask (optional)

    Returns:
        psf: Point Spread Function (normalized intensity)
        psf_image_scale: Spatial scale of PSF in micrometers/pixel
    """
    h, w = wavefront.shape

    # Apply mask to wavefront and replace NaN with 0
    if mask is not None:
        wavefront_masked = np.copy(wavefront)
        # Replace NaN values with 0 (they will be masked out anyway)
        wavefront_masked = np.nan_to_num(wavefront_masked, nan=0.0)
        wavefront_masked = wavefront_masked * mask
    else:
        wavefront_masked = np.nan_to_num(wavefront, nan=0.0)

    # Create complex pupil function
    # P(x,y) = A(x,y) * exp(i * k * W(x,y))
    # where k = 2π/λ
    k = 2 * np.pi / wavelength
    pupil_function = np.exp(1j * k * wavefront_masked)

    # Apply mask to pupil function
    if mask is not None:
        pupil_function = pupil_function * mask

    # Zero-pad for oversampling
    pad_h = (fft_size - h) // 2
    pad_w = (fft_size - w) // 2

    if pad_h >= 0 and pad_w >= 0:
        pupil_padded = np.pad(
            pupil_function,
            ((pad_h, fft_size - h - pad_h), (pad_w, fft_size - w - pad_w)),
            mode='constant',
            constant_values=0
        )
    else:
        # If wavefront is larger than fft_size, crop it
        center_h, center_w = h // 2, w // 2
        crop_h, crop_w = fft_size // 2, fft_size // 2
        pupil_padded = pupil_function[
            center_h - crop_h:center_h + crop_h,
            center_w - crop_w:center_w + crop_w
        ]

    # FFT to get PSF
    psf_complex = np.fft.fft2(pupil_padded)
    psf_complex = np.fft.fftshift(psf_complex)

    # Calculate intensity
    psf = np.abs(psf_complex)**2

    # Normalize to peak = 1
    if np.max(psf) > 0:
        psf = psf / np.max(psf)

    # Calculate spatial scale (µm/pixel in image plane)
    if pixel_size is None:
        pixel_size = pupil_diameter / min(h, w)  # Approximate

    # Scale factor from Fourier optics
    psf_image_scale = wavelength / (pixel_size * fft_size) * 1e6  # in micrometers

    return psf, psf_image_scale


def calculate_strehl_ratio(psf: np.ndarray) -> float:
    """
    Calculate Strehl ratio (peak intensity / diffraction-limited peak).

    For a normalized PSF, this is approximately the peak value.

    Args:
        psf: Point Spread Function (normalized)

    Returns:
        strehl: Strehl ratio (0 to 1)
    """
    # Peak of actual PSF (already normalized to 1 for perfect system)
    peak_actual = np.max(psf)

    # For normalized PSF, Strehl is the peak value
    strehl = peak_actual

    return float(strehl)


def calculate_encircled_energy(
    psf: np.ndarray,
    psf_image_scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate encircled energy as function of radius.

    Args:
        psf: PSF array
        psf_image_scale: Spatial scale (µm/pixel)

    Returns:
        radii: Radii in micrometers
        encircled_energy: Cumulative energy fraction
    """
    h, w = psf.shape
    center_y, center_x = h // 2, w // 2

    # Create distance map from center
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Total energy
    total_energy = np.sum(psf)

    if total_energy == 0:
        return np.array([]), np.array([])

    # Calculate encircled energy for each radius
    max_radius = min(center_x, center_y)
    radii = np.arange(0, max_radius)
    encircled = np.zeros_like(radii, dtype=float)

    for i, r in enumerate(radii):
        mask = distance <= r
        encircled[i] = np.sum(psf[mask]) / total_energy

    # Convert radii to micrometers
    radii_um = radii * psf_image_scale

    return radii_um, encircled


def calculate_psf_fwhm(
    psf: np.ndarray,
    psf_image_scale: float
) -> Tuple[float, float]:
    """
    Calculate Full Width at Half Maximum (FWHM) of PSF.

    Args:
        psf: PSF array
        psf_image_scale: Spatial scale (µm/pixel)

    Returns:
        fwhm_x: FWHM in x direction (micrometers)
        fwhm_y: FWHM in y direction (micrometers)
    """
    h, w = psf.shape
    center_y, center_x = h // 2, w // 2

    # Get central cross-sections
    profile_x = psf[center_y, :]
    profile_y = psf[:, center_x]

    # Find FWHM in x direction
    half_max = np.max(profile_x) / 2
    above_half = profile_x > half_max
    if np.any(above_half):
        indices = np.where(above_half)[0]
        fwhm_x_pixels = len(indices)
        fwhm_x = fwhm_x_pixels * psf_image_scale
    else:
        fwhm_x = 0.0

    # Find FWHM in y direction
    half_max = np.max(profile_y) / 2
    above_half = profile_y > half_max
    if np.any(above_half):
        indices = np.where(above_half)[0]
        fwhm_y_pixels = len(indices)
        fwhm_y = fwhm_y_pixels * psf_image_scale
    else:
        fwhm_y = 0.0

    return fwhm_x, fwhm_y


def get_psf_cross_sections(
    psf: np.ndarray,
    psf_image_scale: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get horizontal and vertical cross-sections through PSF center.

    Args:
        psf: PSF array
        psf_image_scale: Spatial scale (µm/pixel)

    Returns:
        x_coords: X coordinates (micrometers)
        y_coords: Y coordinates (micrometers)
        profile_x: Horizontal profile
        profile_y: Vertical profile
    """
    h, w = psf.shape
    center_y, center_x = h // 2, w // 2

    # Get profiles
    profile_x = psf[center_y, :]
    profile_y = psf[:, center_x]

    # Create coordinate arrays (centered at 0)
    x_coords = (np.arange(w) - center_x) * psf_image_scale
    y_coords = (np.arange(h) - center_y) * psf_image_scale

    return x_coords, y_coords, profile_x, profile_y
