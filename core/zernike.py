"""
Zernike polynomial generation and fitting for wavefront analysis.
"""

import numpy as np
from scipy.special import factorial
from typing import List, Tuple, Optional
from config.settings import DEFAULT_ZERNIKE_MAX_ORDER


def zernike_polynomial(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Calculate Zernike polynomial Z_n^m(rho, theta).

    Args:
        n: Radial order (non-negative integer)
        m: Azimuthal frequency (integer, |m| <= n, n-|m| even)
        rho: Normalized radial coordinate (0 to 1)
        theta: Azimuthal angle (radians)

    Returns:
        Z: Zernike polynomial value
    """
    if (n - abs(m)) % 2 != 0:
        return np.zeros_like(rho)

    # Radial polynomial R_n^m(rho)
    R = radial_polynomial(n, abs(m), rho)

    # Azimuthal component
    if m >= 0:
        Z = R * np.cos(m * theta)
    else:
        Z = R * np.sin(abs(m) * theta)

    return Z


def radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """
    Calculate radial polynomial R_n^m(rho).

    Args:
        n: Radial order
        m: Azimuthal frequency (absolute value)
        rho: Radial coordinate

    Returns:
        R: Radial polynomial value
    """
    R = np.zeros_like(rho, dtype=np.float64)

    for k in range((n - m) // 2 + 1):
        num = (-1)**k * factorial(n - k)
        den = factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        R += (num / den) * rho**(n - 2*k)

    return R


def noll_to_nm(j: int) -> Tuple[int, int]:
    """
    Convert Noll index j to (n, m) Zernike indices.

    Noll ordering is standard in optics (starts at j=1 for piston).

    Args:
        j: Noll index (starting from 1)

    Returns:
        (n, m): Radial order and azimuthal frequency
    """
    n = 0
    j_temp = j
    while j_temp > n + 1:
        n += 1
        j_temp -= n

    m = n - 2 * (j_temp - 1)

    # Determine sign of m based on Noll convention
    if n % 2 == 0:
        m = -m if j_temp % 2 == 0 else m
    else:
        m = m if j_temp % 2 == 0 else -m

    return n, m


def generate_zernike_basis(
    size: int,
    max_order: int = DEFAULT_ZERNIKE_MAX_ORDER,
    mask: Optional[np.ndarray] = None
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Generate Zernike basis functions up to specified order.

    Args:
        size: Image size (assumes square, size x size)
        max_order: Maximum radial order n
        mask: Binary circular mask

    Returns:
        basis: List of Zernike polynomials (2D arrays)
        indices: List of (j, n, m) tuples for each basis function
    """
    # Create normalized coordinates
    y, x = np.ogrid[:size, :size]
    center = size / 2
    x = (x - center) / center
    y = (y - center) / center

    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Apply mask
    if mask is not None:
        rho_masked = np.where(mask, rho, np.nan)
    else:
        rho_masked = rho

    # Generate basis functions
    basis = []
    indices = []

    j = 1  # Noll index
    for n in range(max_order + 1):
        for m_idx in range(-n, n + 1, 2):
            if (n - abs(m_idx)) % 2 == 0:
                Z = zernike_polynomial(n, m_idx, rho_masked, theta)
                basis.append(Z)
                indices.append((j, n, m_idx))
                j += 1

    return basis, indices


class ZernikeFitter:
    """Fit Zernike polynomials to wavefront data."""

    def __init__(self, max_order: int = DEFAULT_ZERNIKE_MAX_ORDER):
        """
        Initialize Zernike fitter.

        Args:
            max_order: Maximum radial order to fit
        """
        self.max_order = max_order
        self.basis = None
        self.indices = None
        self.coefficients = None
        self.excluded_terms = []

    def generate_basis(self, wavefront: np.ndarray, mask: Optional[np.ndarray] = None):
        """
        Generate Zernike basis for given wavefront size.

        Args:
            wavefront: Wavefront data (determines size)
            mask: Binary mask (optional)
        """
        size = wavefront.shape[0]
        self.basis, self.indices = generate_zernike_basis(size, self.max_order, mask)

    def fit(
        self,
        wavefront: np.ndarray,
        mask: Optional[np.ndarray] = None,
        exclude_terms: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Fit Zernike polynomials to wavefront.

        Args:
            wavefront: Wavefront data (2D array)
            mask: Binary mask (optional)
            exclude_terms: List of Noll indices to exclude (e.g., [1, 2, 3])

        Returns:
            coefficients: Zernike coefficients
        """
        if self.basis is None:
            self.generate_basis(wavefront, mask)

        if exclude_terms is None:
            exclude_terms = []
        self.excluded_terms = exclude_terms

        # Prepare data
        if mask is not None:
            valid = mask.astype(bool)
            w_flat = wavefront[valid]
        else:
            valid = np.ones_like(wavefront, dtype=bool)
            w_flat = wavefront.flatten()

        # Remove NaN values
        nan_mask = ~np.isnan(w_flat)
        w_flat = w_flat[nan_mask]

        # Build design matrix (exclude specified terms)
        A_list = []
        for i, (j, n, m) in enumerate(self.indices):
            if j not in exclude_terms:
                if mask is not None:
                    z_flat = self.basis[i][valid]
                else:
                    z_flat = self.basis[i].flatten()

                z_flat = z_flat[nan_mask]
                A_list.append(z_flat)

        if len(A_list) == 0 or len(w_flat) == 0:
            return np.zeros(len(self.indices))

        A = np.column_stack(A_list)

        # Least squares fit
        try:
            coeffs_reduced, _, _, _ = np.linalg.lstsq(A, w_flat, rcond=None)
        except np.linalg.LinAlgError:
            print("Warning: Zernike fitting failed")
            return np.zeros(len(self.indices))

        # Insert zeros for excluded terms
        coeffs_full = []
        reduced_idx = 0
        for j, n, m in self.indices:
            if j in exclude_terms:
                coeffs_full.append(0.0)
            else:
                coeffs_full.append(coeffs_reduced[reduced_idx])
                reduced_idx += 1

        self.coefficients = np.array(coeffs_full)

        return self.coefficients

    def reconstruct(self, coefficients: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reconstruct wavefront from Zernike coefficients.

        Args:
            coefficients: Zernike coefficients (uses fitted if None)

        Returns:
            Reconstructed wavefront
        """
        if coefficients is None:
            coefficients = self.coefficients

        if self.basis is None or coefficients is None:
            return None

        reconstructed = np.zeros_like(self.basis[0])
        for i, coeff in enumerate(coefficients):
            reconstructed += coeff * self.basis[i]

        return reconstructed

    def get_term_names(self) -> List[str]:
        """
        Get descriptive names for Zernike terms.

        Returns:
            List of term names
        """
        names = []
        for j, n, m in self.indices:
            name = zernike_name(n, m)
            names.append(f"Z{j}: {name}")
        return names

    def calculate_residual(
        self,
        wavefront: np.ndarray,
        coefficients: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate residual between wavefront and fit.

        Args:
            wavefront: Original wavefront
            coefficients: Zernike coefficients (uses fitted if None)

        Returns:
            Residual map
        """
        reconstructed = self.reconstruct(coefficients)
        if reconstructed is None:
            return wavefront

        residual = wavefront - reconstructed
        return residual


def zernike_name(n: int, m: int) -> str:
    """
    Get descriptive name for Zernike term.

    Args:
        n: Radial order
        m: Azimuthal frequency

    Returns:
        Descriptive name
    """
    names = {
        (0, 0): "Piston",
        (1, -1): "Tilt Y",
        (1, 1): "Tilt X",
        (2, -2): "Astigmatism 45°",
        (2, 0): "Defocus",
        (2, 2): "Astigmatism 0°",
        (3, -3): "Trefoil Y",
        (3, -1): "Coma Y",
        (3, 1): "Coma X",
        (3, 3): "Trefoil X",
        (4, -4): "Quadrafoil Y",
        (4, -2): "Secondary Astigmatism Y",
        (4, 0): "Spherical",
        (4, 2): "Secondary Astigmatism X",
        (4, 4): "Quadrafoil X",
        (5, -5): "Pentafoil Y",
        (5, -3): "Secondary Trefoil Y",
        (5, -1): "Secondary Coma Y",
        (5, 1): "Secondary Coma X",
        (5, 3): "Secondary Trefoil X",
        (5, 5): "Pentafoil X",
    }
    return names.get((n, m), f"Z({n},{m})")
