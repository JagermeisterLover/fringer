"""
Tab 2: Image Editing and Preprocessing
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QComboBox, QGroupBox, QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from utils.history_manager import HistoryManager
from core.preprocessing import (
    denoise_nonlocal_means, denoise_bilateral, denoise_median,
    smooth_gaussian, smooth_anisotropic, remove_vignette,
    enhance_contrast
)


class ImageViewer(QLabel):
    """Simple image viewer widget."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.setScaledContents(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def display_image(self, image: np.ndarray):
        """Display a numpy array as an image."""
        if image is None:
            return

        h, w = image.shape[:2]

        if len(image.shape) == 2:
            # Grayscale
            q_img = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            # Color (BGR to RGB)
            rgb = np.ascontiguousarray(image[:, :, ::-1])
            q_img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        self.setPixmap(pixmap)


class EditingTab(QWidget):
    """Tab for image editing and preprocessing."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.history = HistoryManager(max_history=20)
        self.current_image = None
        self.mask = None
        self.setup_ui()

    def setup_ui(self):
        """Setup UI for editing tab."""
        layout = QVBoxLayout(self)

        # Top buttons
        button_layout = QHBoxLayout()
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo)
        self.undo_btn.setEnabled(False)
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.redo)
        self.redo_btn.setEnabled(False)
        self.reset_btn = QPushButton("Reset to Original")
        self.reset_btn.clicked.connect(self.reset_to_original)
        self.reset_btn.setEnabled(False)

        button_layout.addWidget(self.undo_btn)
        button_layout.addWidget(self.redo_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        # Main content: image viewer + controls
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Image viewer
        self.image_viewer = ImageViewer()
        splitter.addWidget(self.image_viewer)

        # Right: Controls
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Denoising group
        denoise_group = QGroupBox("Denoising")
        denoise_layout = QVBoxLayout()

        # Denoise method selector
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems(["Non-local Means", "Bilateral", "Median"])
        method_layout.addWidget(self.denoise_combo)
        denoise_layout.addLayout(method_layout)

        # Strength slider
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("Strength:"))
        self.denoise_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_strength_slider.setRange(1, 20)
        self.denoise_strength_slider.setValue(10)
        strength_layout.addWidget(self.denoise_strength_slider)
        self.denoise_strength_label = QLabel("10")
        strength_layout.addWidget(self.denoise_strength_label)
        self.denoise_strength_slider.valueChanged.connect(
            lambda v: self.denoise_strength_label.setText(str(v))
        )
        denoise_layout.addLayout(strength_layout)

        apply_denoise_btn = QPushButton("Apply Denoising")
        apply_denoise_btn.clicked.connect(self.apply_denoising)
        denoise_layout.addWidget(apply_denoise_btn)

        denoise_group.setLayout(denoise_layout)
        control_layout.addWidget(denoise_group)

        # Smoothing group
        smooth_group = QGroupBox("Smoothing")
        smooth_layout = QVBoxLayout()

        # Smooth method selector
        smooth_method_layout = QHBoxLayout()
        smooth_method_layout.addWidget(QLabel("Method:"))
        self.smooth_combo = QComboBox()
        self.smooth_combo.addItems(["Gaussian", "Anisotropic"])
        smooth_method_layout.addWidget(self.smooth_combo)
        smooth_layout.addLayout(smooth_method_layout)

        # Smooth strength slider
        smooth_strength_layout = QHBoxLayout()
        smooth_strength_layout.addWidget(QLabel("Strength:"))
        self.smooth_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.smooth_strength_slider.setRange(1, 50)
        self.smooth_strength_slider.setValue(10)
        smooth_strength_layout.addWidget(self.smooth_strength_slider)
        self.smooth_strength_label = QLabel("10")
        smooth_strength_layout.addWidget(self.smooth_strength_label)
        self.smooth_strength_slider.valueChanged.connect(
            lambda v: self.smooth_strength_label.setText(str(v))
        )
        smooth_layout.addLayout(smooth_strength_layout)

        apply_smooth_btn = QPushButton("Apply Smoothing")
        apply_smooth_btn.clicked.connect(self.apply_smoothing)
        smooth_layout.addWidget(apply_smooth_btn)

        smooth_group.setLayout(smooth_layout)
        control_layout.addWidget(smooth_group)

        # Other operations group
        other_group = QGroupBox("Other Operations")
        other_layout = QVBoxLayout()

        vignette_btn = QPushButton("Remove Vignette")
        vignette_btn.clicked.connect(self.apply_vignette_removal)
        other_layout.addWidget(vignette_btn)

        contrast_btn = QPushButton("Enhance Contrast")
        contrast_btn.clicked.connect(self.apply_contrast_enhancement)
        other_layout.addWidget(contrast_btn)

        other_group.setLayout(other_layout)
        control_layout.addWidget(other_group)

        control_layout.addStretch()

        # Calculate button
        self.calculate_btn = QPushButton("Calculate Phase →")
        self.calculate_btn.clicked.connect(self.calculate_phase)
        self.calculate_btn.setEnabled(False)
        control_layout.addWidget(self.calculate_btn)

        splitter.addWidget(control_widget)
        splitter.setSizes([800, 300])

        layout.addWidget(splitter)

    def set_image(self, image: np.ndarray, mask: np.ndarray):
        """Set the image for editing."""
        self.current_image = image.copy()
        self.mask = mask
        self.history.reset()
        self.history.add_state(self.current_image, "Original")
        self.update_display()
        self.update_buttons()
        self.calculate_btn.setEnabled(True)

    def update_display(self):
        """Update the image display."""
        if self.current_image is not None:
            self.image_viewer.display_image(self.current_image)

    def update_buttons(self):
        """Update button states based on history."""
        self.undo_btn.setEnabled(self.history.can_undo())
        self.redo_btn.setEnabled(self.history.can_redo())
        self.reset_btn.setEnabled(len(self.history.get_history_list()) > 1)

    def undo(self):
        """Undo last operation."""
        image = self.history.undo()
        if image is not None:
            self.current_image = image
            self.update_display()
            self.update_buttons()

    def redo(self):
        """Redo last undone operation."""
        image = self.history.redo()
        if image is not None:
            self.current_image = image
            self.update_display()
            self.update_buttons()

    def reset_to_original(self):
        """Reset to original image."""
        if len(self.history.history) > 0:
            self.current_image = self.history.history[0][0].copy()
            self.history.add_state(self.current_image, "Reset to Original")
            self.update_display()
            self.update_buttons()

    def apply_denoising(self):
        """Apply selected denoising method."""
        if self.current_image is None:
            return

        method = self.denoise_combo.currentText()
        strength = self.denoise_strength_slider.value()

        try:
            if method == "Non-local Means":
                processed = denoise_nonlocal_means(self.current_image, h=strength)
            elif method == "Bilateral":
                processed = denoise_bilateral(self.current_image, sigma_color=strength*5, sigma_space=strength*5)
            elif method == "Median":
                kernel = 3 + (strength // 5) * 2  # 3, 5, 7, 9, etc.
                processed = denoise_median(self.current_image, kernel_size=kernel)
            else:
                return

            self.current_image = processed
            self.history.add_state(self.current_image, f"Denoise: {method}")
            self.update_display()
            self.update_buttons()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Denoising failed: {str(e)}")

    def apply_smoothing(self):
        """Apply selected smoothing method."""
        if self.current_image is None:
            return

        method = self.smooth_combo.currentText()
        strength = self.smooth_strength_slider.value()

        try:
            if method == "Gaussian":
                sigma = strength / 10.0
                processed = smooth_gaussian(self.current_image, sigma=sigma)
            elif method == "Anisotropic":
                processed = smooth_anisotropic(self.current_image, weight=strength/100.0, iterations=strength)
            else:
                return

            self.current_image = processed
            self.history.add_state(self.current_image, f"Smooth: {method}")
            self.update_display()
            self.update_buttons()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Smoothing failed: {str(e)}")

    def apply_vignette_removal(self):
        """Apply vignette removal."""
        if self.current_image is None:
            return

        try:
            processed = remove_vignette(self.current_image)
            self.current_image = processed
            self.history.add_state(self.current_image, "Vignette Removal")
            self.update_display()
            self.update_buttons()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Vignette removal failed: {str(e)}")

    def apply_contrast_enhancement(self):
        """Apply contrast enhancement."""
        if self.current_image is None:
            return

        try:
            processed = enhance_contrast(self.current_image)
            self.current_image = processed
            self.history.add_state(self.current_image, "Contrast Enhancement")
            self.update_display()
            self.update_buttons()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Contrast enhancement failed: {str(e)}")

    def calculate_phase(self):
        """Proceed to analysis tab with processed image."""
        if self.current_image is None:
            return

        # Store processed image in main window
        self.main_window.processed_image = self.current_image

        # Enable analysis tab and switch to it
        self.main_window.enable_tab(2)
        self.main_window.switch_to_tab(2)

        # Initialize analysis tab
        self.main_window.analysis_tab.set_image(self.current_image, self.mask)

        QMessageBox.information(
            self,
            "Success",
            "Image preprocessing complete! Proceed to Analysis tab."
        )
