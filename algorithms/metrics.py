"""
Wavefront quality metrics and calculations.
"""

import numpy as np
from typing import Optional, Dict


def calculate_wavefront_metrics(
    wavefront: np.ndarray,
    mask: Optional[np.ndarray] = None,
    wavelength: float = 632.8e-9,
    unit: str = 'waves'
) -> Dict[str, float]:
    """
    Calculate wavefront quality metrics.

    Args:
        wavefront: Wavefront map in meters
        mask: Binary mask
        wavelength: Wavelength in meters
        unit: Output unit ('waves', 'nm', 'um', 'meters')

    Returns:
        metrics: Dictionary of metrics
    """
    if mask is not None:
        valid = mask.astype(bool)
        valid_data = wavefront[valid]
        # Remove NaN values
        valid_data = valid_data[~np.isnan(valid_data)]
    else:
        valid_data = wavefront.flatten()
        valid_data = valid_data[~np.isnan(valid_data)]

    if len(valid_data) == 0:
        return {
            'rms': 0.0,
            'pv': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'unit': unit
        }

    # Remove piston (mean)
    valid_data = valid_data - np.mean(valid_data)

    # Calculate metrics in meters
    rms = np.sqrt(np.mean(valid_data**2))
    pv = np.max(valid_data) - np.min(valid_data)
    mean = np.mean(valid_data)
    std = np.std(valid_data)
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)

    # Convert to requested unit
    if unit == 'waves':
        conversion = 1.0 / wavelength
    elif unit == 'nm':
        conversion = 1e9
    elif unit == 'um':
        conversion = 1e6
    else:  # meters
        conversion = 1.0

    metrics = {
        'rms': float(rms * conversion),
        'pv': float(pv * conversion),
        'mean': float(mean * conversion),
        'std': float(std * conversion),
        'min': float(min_val * conversion),
        'max': float(max_val * conversion),
        'unit': unit
    }

    return metrics


def calculate_phase_metrics(
    phase: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate phase-specific metrics.

    Args:
        phase: Phase map in radians
        mask: Binary mask

    Returns:
        metrics: Dictionary of metrics
    """
    if mask is not None:
        valid = mask.astype(bool)
        valid_data = phase[valid]
        valid_data = valid_data[~np.isnan(valid_data)]
    else:
        valid_data = phase.flatten()
        valid_data = valid_data[~np.isnan(valid_data)]

    if len(valid_data) == 0:
        return {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'range': 0.0,
            'discontinuities': 0
        }

    # Count phase discontinuities (jumps > π)
    dy, dx = np.gradient(phase)
    if mask is not None:
        dy_valid = dy[valid]
        dx_valid = dx[valid]
    else:
        dy_valid = dy.flatten()
        dx_valid = dx.flatten()

    dy_valid = dy_valid[~np.isnan(dy_valid)]
    dx_valid = dx_valid[~np.isnan(dx_valid)]

    discontinuities = np.sum(np.abs(dy_valid) > np.pi) + np.sum(np.abs(dx_valid) > np.pi)

    metrics = {
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'mean': float(np.mean(valid_data)),
        'std': float(np.std(valid_data)),
        'range': float(np.max(valid_data) - np.min(valid_data)),
        'discontinuities': int(discontinuities)
    }

    return metrics


def calculate_rms_error(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    remove_piston: bool = True,
    remove_tilt: bool = False
) -> float:
    """
    Calculate RMS error.

    Args:
        data: Data array
        mask: Binary mask
        remove_piston: Remove mean (piston)
        remove_tilt: Remove tilt (requires 2D fitting)

    Returns:
        rms: RMS value
    """
    if mask is not None:
        valid = mask.astype(bool)
        valid_data = data[valid]
        valid_data = valid_data[~np.isnan(valid_data)]
    else:
        valid_data = data.flatten()
        valid_data = valid_data[~np.isnan(valid_data)]

    if len(valid_data) == 0:
        return 0.0

    # Remove piston
    if remove_piston:
        valid_data = valid_data - np.mean(valid_data)

    # Remove tilt (requires 2D plane fitting)
    if remove_tilt:
        h, w = data.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        if mask is not None:
            x_valid = x[valid]
            y_valid = y[valid]
            z_valid = data[valid]
        else:
            x_valid = x.flatten()
            y_valid = y.flatten()
            z_valid = data.flatten()

        z_valid = z_valid[~np.isnan(z_valid)]
        x_valid = x_valid[~np.isnan(z_valid)]
        y_valid = y_valid[~np.isnan(z_valid)]

        if len(z_valid) > 0:
            # Fit plane: z = a*x + b*y + c
            A = np.column_stack([x_valid, y_valid, np.ones_like(x_valid)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
                plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]
                data_corrected = data - plane
                if mask is not None:
                    valid_data = data_corrected[valid]
                else:
                    valid_data = data_corrected.flatten()
                valid_data = valid_data[~np.isnan(valid_data)]
            except np.linalg.LinAlgError:
                pass

    rms = np.sqrt(np.mean(valid_data**2))
    return float(rms)


def calculate_pv_error(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Peak-to-Valley error.

    Args:
        data: Data array
        mask: Binary mask

    Returns:
        pv: Peak-to-Valley value
    """
    if mask is not None:
        valid = mask.astype(bool)
        valid_data = data[valid]
        valid_data = valid_data[~np.isnan(valid_data)]
    else:
        valid_data = data.flatten()
        valid_data = valid_data[~np.isnan(valid_data)]

    if len(valid_data) == 0:
        return 0.0

    pv = np.max(valid_data) - np.min(valid_data)
    return float(pv)


def format_metric_value(
    value: float,
    unit: str = 'waves',
    precision: int = 4
) -> str:
    """
    Format metric value with appropriate unit.

    Args:
        value: Metric value
        unit: Unit string
        precision: Number of decimal places

    Returns:
        Formatted string
    """
    if unit == 'waves':
        return f"{value:.{precision}f} λ"
    elif unit == 'nm':
        return f"{value:.{precision}f} nm"
    elif unit == 'um':
        return f"{value:.{precision}f} μm"
    elif unit == 'meters':
        return f"{value:.{precision}e} m"
    else:
        return f"{value:.{precision}f}"
