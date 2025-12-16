"""
Phase unwrapping algorithms for interferogram analysis.
"""

import numpy as np
from skimage.restoration import unwrap_phase
from typing import Optional, Tuple


def unwrap_phase_quality_guided(
    wrapped_phase: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unwrap phase using quality-guided algorithm.

    Args:
        wrapped_phase: Wrapped phase in range [-π, π]
        mask: Binary mask defining valid region

    Returns:
        unwrapped_phase: Continuous phase
        quality_map: Quality metric for each pixel
    """
    # Calculate quality map
    quality_map = calculate_phase_quality(wrapped_phase)

    # Unwrap using scikit-image
    if mask is not None:
        # Don't set masked region to 0 - this creates artificial discontinuities!
        # Instead, use the mask parameter of unwrap_phase if available,
        # or extrapolate into masked regions
        from scipy.ndimage import binary_erosion, binary_dilation

        # Create a slightly eroded mask to avoid edge issues
        mask_bool = mask.astype(bool)

        # Fill masked regions with extrapolated values instead of 0
        # This helps the unwrapping algorithm work correctly
        wrapped_filled = np.copy(wrapped_phase)

        # Simple inpainting: dilate valid region and copy border values
        if not np.all(mask_bool):
            # Get the mean phase in valid region as fallback
            mean_phase = np.mean(wrapped_phase[mask_bool])
            wrapped_filled[~mask_bool] = mean_phase

        # Debug: Check wrapped phase statistics
        print(f"\nPhase Unwrapping Debug:")
        print(f"  Wrapped range: {np.min(wrapped_phase[mask_bool]):.4f} to {np.max(wrapped_phase[mask_bool]):.4f} rad")
        print(f"  Wrapped mean: {np.mean(wrapped_phase[mask_bool]):.4f} rad")

        # Unwrap the filled phase
        unwrapped = unwrap_phase(wrapped_filled)

        # Debug: Check unwrapped phase statistics
        print(f"  Unwrapped range: {np.nanmin(unwrapped[mask_bool]):.4f} to {np.nanmax(unwrapped[mask_bool]):.4f} rad")
        print(f"  Unwrapped span: {np.ptp(unwrapped[mask_bool]):.4f} rad ({np.ptp(unwrapped[mask_bool])/(2*np.pi):.2f} waves)")

        # Set invalid regions to NaN after unwrapping
        unwrapped[~mask_bool] = np.nan
    else:
        unwrapped = unwrap_phase(wrapped_phase)

    return unwrapped, quality_map


def calculate_phase_quality(wrapped_phase: np.ndarray) -> np.ndarray:
    """
    Calculate quality map based on phase derivative variance.
    High quality = low variance in local neighborhood.

    Args:
        wrapped_phase: Wrapped phase array

    Returns:
        Quality map (higher = better quality)
    """
    # Calculate phase derivatives
    dy, dx = np.gradient(wrapped_phase)

    # Wrap derivatives to [-π, π]
    dy = np.arctan2(np.sin(dy), np.cos(dy))
    dx = np.arctan2(np.sin(dx), np.cos(dx))

    # Calculate second derivatives (variance indicator)
    d2y = np.gradient(dy)[0]
    d2x = np.gradient(dx)[1]

    # Quality metric (lower variance = higher quality)
    quality = 1.0 / (1.0 + d2x**2 + d2y**2)

    return quality


def remove_piston_tilt(
    phase: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Remove piston (average) and tilt (linear trend) from phase.

    Args:
        phase: Input phase map
        mask: Binary mask (optional)

    Returns:
        phase_corrected: Phase with piston and tilt removed
    """
    h, w = phase.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    if mask is not None:
        valid = mask.astype(bool)
        x_valid = x[valid]
        y_valid = y[valid]
        z_valid = phase[valid]
    else:
        x_valid = x.flatten()
        y_valid = y.flatten()
        z_valid = phase.flatten()

    # Remove NaN values
    valid_idx = ~np.isnan(z_valid)
    x_valid = x_valid[valid_idx]
    y_valid = y_valid[valid_idx]
    z_valid = z_valid[valid_idx]

    if len(z_valid) == 0:
        return phase

    # Fit plane: z = a*x + b*y + c
    A = np.column_stack([x_valid, y_valid, np.ones_like(x_valid)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)

        # Reconstruct plane
        plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]

        # Subtract plane
        phase_corrected = phase - plane

        if mask is not None:
            phase_corrected[~mask.astype(bool)] = np.nan

        return phase_corrected

    except np.linalg.LinAlgError:
        print("Warning: Piston/tilt removal failed, returning original phase")
        return phase


def phase_to_wavefront(
    unwrapped_phase: np.ndarray,
    wavelength: float = 632.8e-9
) -> np.ndarray:
    """
    Convert phase (radians) to wavefront (meters).

    Args:
        unwrapped_phase: Unwrapped phase in radians
        wavelength: Wavelength in meters (default: 632.8 nm HeNe)

    Returns:
        wavefront: Optical path difference in meters
    """
    wavefront = (wavelength / (2 * np.pi)) * unwrapped_phase
    return wavefront


def wavefront_to_phase(
    wavefront: np.ndarray,
    wavelength: float = 632.8e-9
) -> np.ndarray:
    """
    Convert wavefront (meters) to phase (radians).

    Args:
        wavefront: Optical path difference in meters
        wavelength: Wavelength in meters

    Returns:
        phase: Phase in radians
    """
    phase = (2 * np.pi / wavelength) * wavefront
    return phase
