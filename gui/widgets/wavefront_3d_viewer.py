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
            wavefront: Wavefront data (2D array)
            mask: Binary mask (optional)
        """
        # Clear existing plots
        if self.surface_plot is not None:
            self.view.removeItem(self.surface_plot)

        # Apply mask
        if mask is not None:
            wavefront_display = np.copy(wavefront)
            wavefront_display[~mask.astype(bool)] = np.nan
        else:
            wavefront_display = wavefront

        # Remove NaN for display (replace with minimum value)
        wavefront_clean = wavefront_display.copy()
        valid_data = wavefront_clean[~np.isnan(wavefront_clean)]
        if len(valid_data) > 0:
            min_val = np.nanmin(wavefront_clean)
            wavefront_clean[np.isnan(wavefront_clean)] = min_val

        # Downsample if too large (for performance)
        h, w = wavefront_clean.shape
        if h > 256 or w > 256:
            stride = max(h // 256, w // 256)
            wavefront_clean = wavefront_clean[::stride, ::stride]

        # Convert to wavelength units and scale for visualization
        # Multiply by large factor to make variations visible
        z_data = wavefront_clean * 1e6  # Convert to micrometers

        # Create surface plot
        self.surface_plot = gl.GLSurfacePlotItem(
            z=z_data,
            shader='heightColor',
            computeNormals=True,
            smooth=True
        )

        # Scale x and y to match z
        h_scaled, w_scaled = z_data.shape
        self.surface_plot.scale(1, 1, 0.1)  # Scale z down slightly for better view

        self.view.addItem(self.surface_plot)

    def clear(self):
        """Clear the 3D view."""
        if self.surface_plot is not None:
            self.view.removeItem(self.surface_plot)
            self.surface_plot = None
