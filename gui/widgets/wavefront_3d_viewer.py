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

        # Apply mask - set invalid regions to NaN
        if mask_cropped is not None:
            wavefront_display[~mask_cropped.astype(bool)] = np.nan

        # Check for gradient discontinuities before smoothing (indicates wrapped phase)
        print(f"\nBefore smoothing check:")
        valid_before = wavefront_display[~np.isnan(wavefront_display)]
        if len(valid_before) > 0:
            print(f"  Min: {np.min(valid_before):.3e}, Max: {np.max(valid_before):.3e}")
            print(f"  Mean: {np.mean(valid_before):.3e}, Std: {np.std(valid_before):.3e}")

        # NO SMOOTHING - it's destroying the unwrapped data!
        # Smoothing across boundaries can create artifacts
        wavefront_smooth = wavefront_display.copy()

        # Downsample if too large (for performance) - use simple slicing instead of zoom
        h, w = wavefront_smooth.shape
        if h > 150 or w > 150:
            step = max(h, w) // 150 + 1
            print(f"  Downsampling with step: {step}")
            wavefront_smooth = wavefront_smooth[::step, ::step]
            if mask_cropped is not None:
                mask_display = mask_cropped[::step, ::step]
            else:
                mask_display = None
        else:
            mask_display = mask_cropped

        # For display, replace NaN with mean (not min) to avoid artificial depressions
        wavefront_clean = wavefront_smooth.copy()
        nan_mask = np.isnan(wavefront_clean)
        if np.any(~nan_mask):
            fill_value = np.nanmean(wavefront_clean)
            wavefront_clean[nan_mask] = fill_value
        else:
            wavefront_clean[nan_mask] = 0

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

        # Debug: Print z-data statistics
        print(f"Wavefront 3D Debug:")
        print(f"  Z range (waves): {z_min:.4f} to {z_max:.4f} (range: {z_range:.4f})")
        print(f"  Array shape: {z_data.shape}")
        print(f"  XY size: {max(z_data.shape)}")

        # Create colormap (use jet for better contrast)
        cmap = plt.get_cmap('jet')
        norm = Normalize(vmin=z_min, vmax=z_max)

        # Map z values to colors (vectorized for efficiency)
        colors = cmap(norm(z_data))

        # Ensure colors are RGBA float32 (0-1 range)
        colors = colors.astype(np.float32)

        # Scale z-axis for better 3D visualization
        # Need AGGRESSIVE scaling since wavefront variations are small (typically 0.01-0.1 waves)
        xy_size = max(z_data.shape)
        if z_range > 0.001:  # Only scale if there's meaningful variation
            # Make z-variations very prominent: use 50-100x amplification
            # For 0.1 wave range across 100 pixel surface, this gives z-height of 20-40 units
            z_scale = xy_size / (z_range * 0.5)  # Aggressive scaling
            # Cap the scaling to avoid extreme values
            z_scale = min(z_scale, xy_size * 100)
        else:
            z_scale = xy_size  # Default scaling

        print(f"  Z scale factor: {z_scale:.2f}")

        z_data_scaled = z_data * z_scale

        print(f"  Scaled Z range: {np.min(z_data_scaled):.2f} to {np.max(z_data_scaled):.2f}")

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
