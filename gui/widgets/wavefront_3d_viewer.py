"""
3D wavefront visualization widget using PyQtGraph.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph.opengl as gl
from config.settings import VIEW_DISTANCE, VIEW_ELEVATION, VIEW_AZIMUTH


class Wavefront3DViewer(QWidget):
    """3D viewer for wavefront data using PyQtGraph OpenGL."""

    def __init__(self):
        super().__init__()
        self.view = gl.GLViewWidget()
        self.surface_plot = None

        # Setup layout
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Configure view
        self.view.setCameraPosition(distance=VIEW_DISTANCE, elevation=VIEW_ELEVATION, azimuth=VIEW_AZIMUTH)
        self.view.opts['bgcolor'] = (20, 20, 20, 255)

        # Add grid
        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        self.view.addItem(grid)

    def display_wavefront(self, wavefront: np.ndarray, mask: np.ndarray = None):
        """
        Display wavefront as 3D surface.

        Args:
            wavefront: Wavefront data (2D array) in meters
            mask: Binary mask (optional)
        """
        # Clear existing plots
        if self.surface_plot is not None:
            self.view.removeItem(self.surface_plot)

        # Apply mask and get valid data
        if mask is not None:
            valid = mask.astype(bool)
            valid_data = wavefront[valid]
        else:
            valid_data = wavefront.flatten()

        # Remove NaN values
        valid_data = valid_data[~np.isnan(valid_data)]

        if len(valid_data) == 0:
            return

        # Remove piston (mean) and optionally tilt for better visualization
        mean_val = np.mean(valid_data)

        # Create display array
        wavefront_display = np.copy(wavefront)
        wavefront_display = wavefront_display - mean_val

        # Smooth the wavefront data slightly to reduce noise artifacts
        from scipy.ndimage import gaussian_filter
        wavefront_smooth = gaussian_filter(wavefront_display, sigma=1.0)

        # Apply mask for display
        if mask is not None:
            wavefront_smooth[~mask.astype(bool)] = np.nan

        # Downsample if too large (for performance) - use average instead of simple indexing
        h, w = wavefront_smooth.shape
        if h > 150 or w > 150:
            from scipy.ndimage import zoom
            scale_factor = 150.0 / max(h, w)
            wavefront_smooth = zoom(wavefront_smooth, scale_factor, order=3)
            if mask is not None:
                mask_display = zoom(mask.astype(float), scale_factor, order=1) > 0.5
            else:
                mask_display = None
        else:
            mask_display = mask

        # Replace NaN with boundary values for smoother display
        wavefront_clean = wavefront_smooth.copy()
        if mask_display is not None:
            # Fill masked regions with edge values instead of zero
            valid_mean = np.nanmean(wavefront_clean[mask_display.astype(bool)])
            wavefront_clean[~mask_display.astype(bool)] = valid_mean if not np.isnan(valid_mean) else 0
        wavefront_clean[np.isnan(wavefront_clean)] = 0

        # Convert to wavelengths for display (more intuitive than meters)
        # Using HeNe wavelength 632.8nm
        wavelength = 632.8e-9
        z_data = wavefront_clean / wavelength

        # Scale for better 3D visualization
        valid_z = z_data[np.isfinite(z_data)]
        if len(valid_z) > 0:
            z_range = np.ptp(valid_z)
            if z_range > 0:
                # Scale to range of approximately 20-30 units for better 3D view
                scale_factor = 25.0 / max(z_range, 0.01)
                z_data = z_data * scale_factor

        # Create surface plot with height-based coloring
        self.surface_plot = gl.GLSurfacePlotItem(
            z=z_data,
            shader='heightColor',
            computeNormals=True,
            smooth=True,
            drawEdges=False,
            glOptions='opaque'
        )

        # Set color map (built-in heightColor shader)
        # The shader automatically colors based on height

        # Center and scale the plot
        h_scaled, w_scaled = z_data.shape
        self.surface_plot.translate(-w_scaled/2, -h_scaled/2, 0)
        self.surface_plot.scale(1, 1, 1)

        self.view.addItem(self.surface_plot)

    def clear(self):
        """Clear the 3D view."""
        if self.surface_plot is not None:
            self.view.removeItem(self.surface_plot)
            self.surface_plot = None
