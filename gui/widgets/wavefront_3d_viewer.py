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

        # Remove piston (mean) from valid data only
        mean_val = np.mean(valid_data)

        # Create display array
        wavefront_display = np.copy(wavefront)
        wavefront_display = wavefront_display - mean_val

        # Apply mask for display
        if mask is not None:
            wavefront_display[~mask.astype(bool)] = np.nan

        # Downsample if too large (for performance)
        h, w = wavefront_display.shape
        if h > 200 or w > 200:
            stride = max(h // 200, w // 200)
            wavefront_display = wavefront_display[::stride, ::stride]
            if mask is not None:
                mask_display = mask[::stride, ::stride]
            else:
                mask_display = None
        else:
            mask_display = mask

        # Replace NaN with mean for display
        wavefront_clean = wavefront_display.copy()
        wavefront_clean[np.isnan(wavefront_clean)] = 0

        # Convert to micrometers for better scale
        z_data = wavefront_clean * 1e6

        # Normalize z range to reasonable visualization scale
        valid_z = z_data[np.isfinite(z_data)]
        if len(valid_z) > 0:
            z_range = np.ptp(valid_z)
            if z_range > 0:
                # Scale to range of approximately 10-20 units for better 3D view
                scale_factor = 15.0 / z_range
                z_data = z_data * scale_factor

        # Create surface plot
        colors = np.zeros((*z_data.shape, 4))

        # Color based on height
        if len(valid_z) > 0:
            z_min, z_max = np.nanmin(z_data), np.nanmax(z_data)
            if z_max > z_min:
                normalized = (z_data - z_min) / (z_max - z_min)
                # Color map: blue (low) -> green (mid) -> red (high)
                colors[..., 2] = np.clip(1 - normalized, 0, 1)  # Blue
                colors[..., 1] = np.clip(1 - np.abs(normalized - 0.5) * 2, 0, 1)  # Green
                colors[..., 0] = np.clip(normalized, 0, 1)  # Red
                colors[..., 3] = 1.0  # Alpha

        self.surface_plot = gl.GLSurfacePlotItem(
            z=z_data,
            colors=colors,
            shader='shaded',
            computeNormals=True,
            smooth=True,
            drawEdges=False
        )

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
