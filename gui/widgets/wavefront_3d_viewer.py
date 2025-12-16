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
        self.view.opts['bgcolor'] = (240, 240, 245, 255)  # Light gray background

        # Add grid
        self.grid = gl.GLGridItem()
        self.grid.scale(10, 10, 1)
        self.grid.setColor((100, 100, 100, 100))  # Dark gray grid
        self.view.addItem(self.grid)

        # Add axis
        self.axis = gl.GLAxisItem()
        self.axis.setSize(50, 50, 50)
        self.view.addItem(self.axis)

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

        # Crop to valid region if mask is provided
        if mask is not None:
            # Find bounding box of valid mask region
            rows, cols = np.where(mask.astype(bool))
            if len(rows) == 0 or len(cols) == 0:
                return

            row_min, row_max = rows.min(), rows.max() + 1
            col_min, col_max = cols.min(), cols.max() + 1

            # Add small padding (5% on each side)
            padding = int(0.05 * max(row_max - row_min, col_max - col_min))
            row_min = max(0, row_min - padding)
            row_max = min(wavefront.shape[0], row_max + padding)
            col_min = max(0, col_min - padding)
            col_max = min(wavefront.shape[1], col_max + padding)

            # Crop to bounding box
            wavefront_cropped = wavefront[row_min:row_max, col_min:col_max]
            mask_cropped = mask[row_min:row_max, col_min:col_max]
        else:
            wavefront_cropped = wavefront
            mask_cropped = None

        # Get valid data for statistics
        if mask_cropped is not None:
            valid_data = wavefront_cropped[mask_cropped.astype(bool)]
        else:
            valid_data = wavefront_cropped.flatten()

        # Remove NaN values
        valid_data = valid_data[~np.isnan(valid_data)]

        if len(valid_data) == 0:
            return

        # Remove piston (mean) for better visualization
        mean_val = np.mean(valid_data)

        # Create display array
        wavefront_display = np.copy(wavefront_cropped)
        wavefront_display = wavefront_display - mean_val

        # Smooth the wavefront data slightly to reduce noise artifacts
        from scipy.ndimage import gaussian_filter
        wavefront_smooth = gaussian_filter(wavefront_display, sigma=1.0)

        # Apply mask for display
        if mask_cropped is not None:
            wavefront_smooth[~mask_cropped.astype(bool)] = np.nan

        # Downsample if too large (for performance)
        h, w = wavefront_smooth.shape
        if h > 150 or w > 150:
            from scipy.ndimage import zoom
            scale_factor = 150.0 / max(h, w)
            wavefront_smooth = zoom(wavefront_smooth, scale_factor, order=3)
            if mask_cropped is not None:
                mask_display = zoom(mask_cropped.astype(float), scale_factor, order=1) > 0.5
            else:
                mask_display = None
        else:
            mask_display = mask_cropped

        # Replace NaN with minimum value for smoother display edges
        wavefront_clean = wavefront_smooth.copy()
        if mask_display is not None:
            # Get the valid data range
            valid_clean = wavefront_clean[mask_display.astype(bool)]
            valid_clean = valid_clean[~np.isnan(valid_clean)]
            if len(valid_clean) > 0:
                edge_value = np.min(valid_clean)
            else:
                edge_value = 0
            # Fill masked regions with edge value
            wavefront_clean[~mask_display.astype(bool)] = edge_value
        wavefront_clean[np.isnan(wavefront_clean)] = 0

        # Convert to wavelengths for display (more intuitive than meters)
        # Using HeNe wavelength 632.8nm
        wavelength = 632.8e-9
        z_data = wavefront_clean / wavelength

        # Create explicit color map based on height
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        # Get valid z range for colormap
        valid_z = z_data[np.isfinite(z_data)]
        if len(valid_z) == 0:
            return

        z_min, z_max = np.min(valid_z), np.max(valid_z)
        z_range = z_max - z_min

        # Create colormap (use jet for better contrast)
        cmap = plt.get_cmap('jet')
        norm = Normalize(vmin=z_min, vmax=z_max)

        # Map z values to colors (vectorized for efficiency)
        colors = cmap(norm(z_data))

        # Scale z-axis for better 3D visualization
        # Make z-scale proportional to xy dimensions
        xy_size = max(z_data.shape)
        if z_range > 0:
            # Scale so z variations are visible but not exaggerated
            z_scale = xy_size / (5 * z_range)  # z will be 1/5 of xy size for 1 wave variation
        else:
            z_scale = 1.0

        z_data_scaled = z_data * z_scale

        # Create surface plot with explicit colors
        self.surface_plot = gl.GLSurfacePlotItem(
            z=z_data_scaled,
            colors=colors,
            computeNormals=True,
            smooth=True,
            drawEdges=False,
            glOptions='opaque'
        )

        # Center the plot
        h_scaled, w_scaled = z_data.shape
        self.surface_plot.translate(-w_scaled/2, -h_scaled/2, 0)

        # Update grid to match surface size
        grid_size = max(h_scaled, w_scaled) / 10
        self.grid.resetTransform()
        self.grid.scale(grid_size, grid_size, 1)

        self.view.addItem(self.surface_plot)

    def clear(self):
        """Clear the 3D view."""
        if self.surface_plot is not None:
            self.view.removeItem(self.surface_plot)
            self.surface_plot = None
