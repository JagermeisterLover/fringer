"""
Tab 3: Analysis and Results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTabWidget, QGroupBox, QMessageBox, QTextEdit, QComboBox
)
from PyQt6.QtCore import Qt
from core.phase_extraction import extract_phase_fft
from core.phase_unwrapping import (
    unwrap_phase_quality_guided, remove_piston_tilt, phase_to_wavefront
)
from core.zernike import ZernikeFitter
from core.psf_calculator import calculate_psf, calculate_strehl_ratio
from algorithms.metrics import calculate_wavefront_metrics, calculate_phase_metrics, format_metric_value
from gui.widgets.wavefront_3d_viewer import Wavefront3DViewer
from config.settings import DEFAULT_WAVELENGTH, DEFAULT_ZERNIKE_MAX_ORDER, DEFAULT_PUPIL_DIAMETER


class MatplotlibCanvas(FigureCanvas):
    """Canvas for embedding matplotlib figures."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


class AnalysisTab(QWidget):
    """Tab for phase analysis and results."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.image = None
        self.mask = None

        # Analysis results
        self.wrapped_phase = None
        self.unwrapped_phase = None
        self.wavefront = None
        self.psf = None
        self.psf_scale = None
        self.zernike_fitter = None

        self.setup_ui()

    def setup_ui(self):
        """Setup UI for analysis tab."""
        layout = QVBoxLayout(self)

        # Control buttons at top
        button_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("Run Phase Analysis")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setEnabled(False)
        button_layout.addWidget(self.analyze_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        # Nested tabs for different visualizations
        self.result_tabs = QTabWidget()

        # Create sub-tabs
        self.create_phase_tab()
        self.create_wavefront_tab()
        self.create_psf_tab()
        self.create_zernike_tab()

        layout.addWidget(self.result_tabs)

    def create_phase_tab(self):
        """Create phase visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Canvas for phase display
        self.phase_canvas = MatplotlibCanvas(tab, width=6, height=5)
        layout.addWidget(self.phase_canvas)

        # Metrics display
        metrics_group = QGroupBox("Phase Metrics")
        metrics_layout = QVBoxLayout()
        self.phase_metrics_label = QLabel("Run analysis to see metrics")
        metrics_layout.addWidget(self.phase_metrics_label)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        self.result_tabs.addTab(tab, "Phase Maps")

    def create_wavefront_tab(self):
        """Create wavefront visualization tab."""
        from PyQt6.QtWidgets import QSplitter

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        # Use splitter for proper proportions
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: 3D view (just the viewer, no extra label)
        self.wavefront_3d_viewer = Wavefront3DViewer()
        self.wavefront_3d_viewer.setMinimumWidth(400)
        splitter.addWidget(self.wavefront_3d_viewer)

        # Right: Metrics (compact)
        metrics_widget = QWidget()
        metrics_widget.setMaximumWidth(350)
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.setContentsMargins(5, 5, 5, 5)

        metrics_group = QGroupBox("Wavefront Metrics")
        metrics_box_layout = QVBoxLayout()
        self.wavefront_metrics_label = QTextEdit()
        self.wavefront_metrics_label.setReadOnly(True)
        self.wavefront_metrics_label.setMaximumHeight(250)
        self.wavefront_metrics_label.setText("Run analysis to see metrics")
        metrics_box_layout.addWidget(self.wavefront_metrics_label)
        metrics_group.setLayout(metrics_box_layout)
        metrics_layout.addWidget(metrics_group)
        metrics_layout.addStretch()
        splitter.addWidget(metrics_widget)

        # Set initial sizes: 80% for 3D view, 20% for metrics
        splitter.setSizes([800, 200])
        splitter.setStretchFactor(0, 3)  # 3D view gets more stretch
        splitter.setStretchFactor(1, 1)  # Metrics gets less

        layout.addWidget(splitter)
        self.result_tabs.addTab(tab, "3D Wavefront")

    def create_psf_tab(self):
        """Create PSF visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Canvas for PSF display
        self.psf_canvas = MatplotlibCanvas(tab, width=6, height=5)
        layout.addWidget(self.psf_canvas)

        # Metrics display
        metrics_group = QGroupBox("PSF Metrics")
        metrics_layout = QVBoxLayout()
        self.psf_metrics_label = QLabel("Run analysis to see metrics")
        metrics_layout.addWidget(self.psf_metrics_label)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        self.result_tabs.addTab(tab, "PSF")

    def create_zernike_tab(self):
        """Create Zernike analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Controls
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Max Order:"))
        self.zernike_order_combo = QComboBox()
        self.zernike_order_combo.addItems(["4", "6", "8", "10", "12"])
        self.zernike_order_combo.setCurrentText("6")
        control_layout.addWidget(self.zernike_order_combo)

        fit_btn = QPushButton("Fit Zernike")
        fit_btn.clicked.connect(self.fit_zernike)
        control_layout.addWidget(fit_btn)
        control_layout.addStretch()
        layout.addLayout(control_layout)

        # Canvas for coefficient display
        self.zernike_canvas = MatplotlibCanvas(tab, width=6, height=5)
        layout.addWidget(self.zernike_canvas)

        # Coefficients display
        coeff_group = QGroupBox("Zernike Coefficients")
        coeff_layout = QVBoxLayout()
        self.zernike_coeffs_text = QTextEdit()
        self.zernike_coeffs_text.setReadOnly(True)
        self.zernike_coeffs_text.setText("Fit Zernike polynomials to see coefficients")
        coeff_layout.addWidget(self.zernike_coeffs_text)
        coeff_group.setLayout(coeff_layout)
        layout.addWidget(coeff_group)

        self.result_tabs.addTab(tab, "Zernike Analysis")

    def set_image(self, image: np.ndarray, mask: np.ndarray):
        """Set the image for analysis."""
        self.image = image
        self.mask = mask
        self.analyze_btn.setEnabled(True)

    def run_analysis(self):
        """Run phase extraction and analysis."""
        if self.image is None:
            return

        try:
            # 1. Extract phase
            self.wrapped_phase, fft_spectrum = extract_phase_fft(self.image, self.mask)

            # 2. Unwrap phase
            self.unwrapped_phase, quality_map = unwrap_phase_quality_guided(self.wrapped_phase, self.mask)

            # 3. Remove piston and tilt
            self.unwrapped_phase = remove_piston_tilt(self.unwrapped_phase, self.mask)

            # 4. Convert to wavefront
            self.wavefront = phase_to_wavefront(self.unwrapped_phase, DEFAULT_WAVELENGTH)

            # 5. Calculate PSF
            self.psf, self.psf_scale = calculate_psf(
                self.wavefront,
                wavelength=DEFAULT_WAVELENGTH,
                pupil_diameter=DEFAULT_PUPIL_DIAMETER,
                mask=self.mask
            )

            # Update visualizations
            self.display_phase_maps()
            self.display_wavefront()
            self.display_psf()

            # Store results in main window
            self.main_window.wrapped_phase = self.wrapped_phase
            self.main_window.unwrapped_phase = self.unwrapped_phase
            self.main_window.wavefront = self.wavefront

            QMessageBox.information(
                self,
                "Success",
                "Phase analysis complete! Explore the different visualization tabs."
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")

    def display_phase_maps(self):
        """Display wrapped and unwrapped phase maps."""
        self.phase_canvas.ax.clear()

        # Create subplots
        self.phase_canvas.fig.clear()
        ax1 = self.phase_canvas.fig.add_subplot(121)
        ax2 = self.phase_canvas.fig.add_subplot(122)

        # Wrapped phase
        im1 = ax1.imshow(self.wrapped_phase, cmap='twilight')
        ax1.set_title('Wrapped Phase')
        ax1.axis('off')
        self.phase_canvas.fig.colorbar(im1, ax=ax1, fraction=0.046)

        # Unwrapped phase
        unwrapped_display = self.unwrapped_phase.copy()
        if self.mask is not None:
            unwrapped_display[~self.mask.astype(bool)] = np.nan

        im2 = ax2.imshow(unwrapped_display, cmap='jet')
        ax2.set_title('Unwrapped Phase')
        ax2.axis('off')
        self.phase_canvas.fig.colorbar(im2, ax=ax2, fraction=0.046)

        self.phase_canvas.fig.tight_layout()
        self.phase_canvas.draw()

        # Display metrics
        metrics = calculate_phase_metrics(self.unwrapped_phase, self.mask)
        metrics_text = f"""
        Min: {metrics['min']:.3f} rad
        Max: {metrics['max']:.3f} rad
        Mean: {metrics['mean']:.3f} rad
        Std Dev: {metrics['std']:.3f} rad
        Range: {metrics['range']:.3f} rad
        """
        self.phase_metrics_label.setText(metrics_text)

    def display_wavefront(self):
        """Display 3D wavefront."""
        # Debug: Check wavefront statistics before display
        if self.mask is not None:
            valid_wf = self.wavefront[self.mask.astype(bool)]
        else:
            valid_wf = self.wavefront.flatten()
        valid_wf = valid_wf[~np.isnan(valid_wf)]

        print(f"\nWavefront before 3D display:")
        print(f"  Shape: {self.wavefront.shape}")
        print(f"  Valid points: {len(valid_wf)}")
        print(f"  Range (meters): {np.min(valid_wf):.3e} to {np.max(valid_wf):.3e}")
        print(f"  Range (waves @ 632.8nm): {np.min(valid_wf)/632.8e-9:.4f} to {np.max(valid_wf)/632.8e-9:.4f}")
        print(f"  Std dev (waves): {np.std(valid_wf)/632.8e-9:.4f}")

        # Check for wrapped phase characteristics (sudden jumps)
        if len(valid_wf) > 100:
            wf_waves = valid_wf / 632.8e-9
            diffs = np.diff(sorted(wf_waves))
            max_diff = np.max(diffs)
            print(f"  Max consecutive value diff: {max_diff:.4f} waves")
            if max_diff > 0.5:
                print(f"  WARNING: Large jumps detected - may be wrapped phase!")

        self.wavefront_3d_viewer.display_wavefront(self.wavefront, self.mask)

        # Display metrics
        metrics = calculate_wavefront_metrics(self.wavefront, self.mask, DEFAULT_WAVELENGTH, unit='waves')

        metrics_text = f"""
<b>Wavefront Quality Metrics:</b>

RMS: {format_metric_value(metrics['rms'], 'waves', 4)}
Peak-to-Valley: {format_metric_value(metrics['pv'], 'waves', 4)}

Mean: {format_metric_value(metrics['mean'], 'waves', 4)}
Std Dev: {format_metric_value(metrics['std'], 'waves', 4)}

Min: {format_metric_value(metrics['min'], 'waves', 4)}
Max: {format_metric_value(metrics['max'], 'waves', 4)}
        """

        self.wavefront_metrics_label.setHtml(metrics_text)

    def display_psf(self):
        """Display PSF."""
        self.psf_canvas.ax.clear()

        # Display PSF in log scale
        psf_log = np.log10(self.psf + 1e-10)

        im = self.psf_canvas.ax.imshow(psf_log, cmap='hot')
        self.psf_canvas.ax.set_title('Point Spread Function (log scale)')
        self.psf_canvas.ax.axis('off')
        self.psf_canvas.fig.colorbar(im, ax=self.psf_canvas.ax)

        self.psf_canvas.fig.tight_layout()
        self.psf_canvas.draw()

        # Calculate and display Strehl ratio
        strehl = calculate_strehl_ratio(self.psf)

        metrics_text = f"""
        Strehl Ratio: {strehl:.4f}
        PSF Scale: {self.psf_scale:.3f} μm/pixel
        """
        self.psf_metrics_label.setText(metrics_text)

    def fit_zernike(self):
        """Fit Zernike polynomials to wavefront."""
        if self.wavefront is None:
            QMessageBox.warning(self, "Warning", "Please run phase analysis first.")
            return

        try:
            max_order = int(self.zernike_order_combo.currentText())

            # Create fitter
            self.zernike_fitter = ZernikeFitter(max_order=max_order)

            # Fit
            coefficients = self.zernike_fitter.fit(self.wavefront, self.mask)

            # Store in main window
            self.main_window.zernike_coefficients = coefficients

            # Display results
            self.display_zernike_results(coefficients)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Zernike fitting failed: {str(e)}")

    def display_zernike_results(self, coefficients: np.ndarray):
        """Display Zernike fitting results."""
        # Plot bar chart
        self.zernike_canvas.ax.clear()

        indices = np.arange(len(coefficients))
        self.zernike_canvas.ax.bar(indices, coefficients)
        self.zernike_canvas.ax.set_xlabel('Zernike Term (Noll Index)')
        self.zernike_canvas.ax.set_ylabel('Coefficient (waves)')
        self.zernike_canvas.ax.set_title('Zernike Coefficients')
        self.zernike_canvas.ax.grid(True, alpha=0.3)

        self.zernike_canvas.fig.tight_layout()
        self.zernike_canvas.draw()

        # Display coefficient values
        term_names = self.zernike_fitter.get_term_names()
        coeff_text = "<b>Zernike Coefficients (in waves):</b><br><br>"

        for i, (name, coeff) in enumerate(zip(term_names, coefficients)):
            coeff_text += f"{name}: {coeff:.6f}<br>"

        self.zernike_coeffs_text.setHtml(coeff_text)
