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
        wrapped_phase: Wrapped phase in range [-π, π] (NaN for invalid regions)
        mask: Binary mask defining valid region (optional, will be derived from NaN if not provided)

    Returns:
        unwrapped_phase: Continuous phase
        quality_map: Quality metric for each pixel
    """
    # Determine valid regions from NaN values or mask
    if mask is not None:
        mask_bool = mask.astype(bool)
    else:
        mask_bool = ~np.isnan(wrapped_phase)

    # Replace NaN with 0 for unwrapping
    # Don't interpolate - that smooths out the 2π discontinuities we need to detect!
    wrapped_filled = np.nan_to_num(wrapped_phase, nan=0.0)

    # Debug: Check wrapped phase statistics
    print(f"\nPhase Unwrapping Debug:")
    if np.any(mask_bool):
        valid_wrapped = wrapped_phase[mask_bool]
        valid_wrapped = valid_wrapped[~np.isnan(valid_wrapped)]
        if len(valid_wrapped) > 0:
            print(f"  Wrapped range: {np.min(valid_wrapped):.4f} to {np.max(valid_wrapped):.4f} rad")
            print(f"  Wrapped mean: {np.mean(valid_wrapped):.4f} rad")
            print(f"  Valid pixels: {len(valid_wrapped)}")
            # Check for phase jumps
            sorted_phase = np.sort(valid_wrapped)
            phase_diff = np.diff(sorted_phase)
            max_jump = np.max(phase_diff)
            print(f"  Max phase jump: {max_jump:.4f} rad ({max_jump/np.pi:.2f}π)")

    # Try using scikit-image unwrap_phase (quality-guided)
    try:
        unwrapped = unwrap_phase(wrapped_filled)
        print(f"  Using scikit-image unwrap_phase")
    except Exception as e:
        print(f"  scikit-image unwrap failed: {e}, trying numpy unwrap")
        # Fallback to numpy unwrap (row-by-row then column-by-column)
        unwrapped = np.copy(wrapped_filled)

        # Unwrap along rows first
        for i in range(unwrapped.shape[0]):
            unwrapped[i, :] = np.unwrap(unwrapped[i, :])

        # Then unwrap along columns
        for j in range(unwrapped.shape[1]):
            unwrapped[:, j] = np.unwrap(unwrapped[:, j])

    # Debug: Check unwrapped phase statistics
    if np.any(mask_bool):
        valid_unwrapped = unwrapped[mask_bool]
        valid_unwrapped = valid_unwrapped[~np.isnan(valid_unwrapped)]
        if len(valid_unwrapped) > 0:
            print(f"  Unwrapped range: {np.min(valid_unwrapped):.4f} to {np.max(valid_unwrapped):.4f} rad")
            print(f"  Unwrapped span: {np.ptp(valid_unwrapped):.4f} rad ({np.ptp(valid_unwrapped)/(2*np.pi):.2f} waves)")

    # Set invalid regions back to NaN after unwrapping
    unwrapped[~mask_bool] = np.nan

    # Calculate quality map
    quality_map = calculate_phase_quality(wrapped_phase)

    return unwrapped, quality_map


def calculate_phase_quality(wrapped_phase: np.ndarray) -> np.ndarray:
    """
    Calculate quality map based on phase derivative variance.
    High quality = low variance in local neighborhood.

    Args:
        wrapped_phase: Wrapped phase array (may contain NaN)

    Returns:
        Quality map (higher = better quality)
    """
    # Replace NaN with 0 for gradient calculation
    phase_clean = np.nan_to_num(wrapped_phase, nan=0.0)

    # Calculate phase derivatives
    dy, dx = np.gradient(phase_clean)

    # Wrap derivatives to [-π, π]
    dy = np.arctan2(np.sin(dy), np.cos(dy))
    dx = np.arctan2(np.sin(dx), np.cos(dx))

    # Calculate second derivatives (variance indicator)
    d2y = np.gradient(dy)[0]
    d2x = np.gradient(dx)[1]

    # Quality metric (lower variance = higher quality)
    quality = 1.0 / (1.0 + d2x**2 + d2y**2)

    # Set quality to 0 where phase is NaN
    quality[np.isnan(wrapped_phase)] = 0.0

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
