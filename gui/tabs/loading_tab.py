"""
Tab 1: Image Loading and Masking
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSpinBox, QCheckBox, QGroupBox, QSplitter,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt
from gui.widgets.mask_editor import MaskEditor
from core.image_loader import load_image, get_image_info
from core.masking import auto_detect_circle
from config.settings import IMAGE_FILE_FILTER


class LoadingTab(QWidget):
    """Tab for loading images and defining masks."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.image = None
        self.mask = None
        self.setup_ui()

    def setup_ui(self):
        """Setup UI for loading tab."""
        layout = QVBoxLayout(self)

        # Top buttons
        button_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        self.auto_mask_btn = QPushButton("Auto-detect Mask")
        self.auto_mask_btn.clicked.connect(self.auto_detect_mask)
        self.auto_mask_btn.setEnabled(False)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_mask)
        self.reset_btn.setEnabled(False)

        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.auto_mask_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        # Main content: image viewer + controls
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Image viewer with mask editor
        self.mask_editor = MaskEditor()
        self.mask_editor.maskChanged.connect(self.on_mask_changed)
        splitter.addWidget(self.mask_editor)

        # Right: Controls
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Mask parameters group
        mask_group = QGroupBox("Mask Parameters")
        mask_layout = QVBoxLayout()

        # Use outer mask checkbox
        self.use_outer_mask_check = QCheckBox("Use Outer Mask")
        self.use_outer_mask_check.setChecked(True)
        self.use_outer_mask_check.toggled.connect(self.on_outer_mask_toggled)
        mask_layout.addWidget(self.use_outer_mask_check)

        # Outer radius
        outer_layout = QHBoxLayout()
        outer_layout.addWidget(QLabel("  Outer Radius:"))
        self.outer_radius_spin = QSpinBox()
        self.outer_radius_spin.setRange(0, 4000)
        self.outer_radius_spin.setValue(200)
        self.outer_radius_spin.valueChanged.connect(self.update_mask_from_controls)
        outer_layout.addWidget(self.outer_radius_spin)
        mask_layout.addLayout(outer_layout)

        # Use inner mask checkbox
        self.use_inner_mask_check = QCheckBox("Use Inner Mask (Central Obscuration)")
        self.use_inner_mask_check.setChecked(True)
        self.use_inner_mask_check.toggled.connect(self.on_inner_mask_toggled)
        mask_layout.addWidget(self.use_inner_mask_check)

        # Inner radius
        inner_layout = QHBoxLayout()
        inner_layout.addWidget(QLabel("  Inner Radius:"))
        self.inner_radius_spin = QSpinBox()
        self.inner_radius_spin.setRange(0, 4000)
        self.inner_radius_spin.setValue(50)
        self.inner_radius_spin.valueChanged.connect(self.update_mask_from_controls)
        inner_layout.addWidget(self.inner_radius_spin)
        mask_layout.addLayout(inner_layout)

        # Center X
        cx_layout = QHBoxLayout()
        cx_layout.addWidget(QLabel("Center X:"))
        self.center_x_spin = QSpinBox()
        self.center_x_spin.setRange(0, 4000)
        self.center_x_spin.setValue(512)
        self.center_x_spin.valueChanged.connect(self.update_mask_from_controls)
        cx_layout.addWidget(self.center_x_spin)
        mask_layout.addLayout(cx_layout)

        # Center Y
        cy_layout = QHBoxLayout()
        cy_layout.addWidget(QLabel("Center Y:"))
        self.center_y_spin = QSpinBox()
        self.center_y_spin.setRange(0, 4000)
        self.center_y_spin.setValue(512)
        self.center_y_spin.valueChanged.connect(self.update_mask_from_controls)
        cy_layout.addWidget(self.center_y_spin)
        mask_layout.addLayout(cy_layout)

        # Show mask checkbox
        self.show_mask_check = QCheckBox("Show Mask")
        self.show_mask_check.setChecked(True)
        self.show_mask_check.toggled.connect(self.mask_editor.set_mask_visible)
        mask_layout.addWidget(self.show_mask_check)

        mask_group.setLayout(mask_layout)
        control_layout.addWidget(mask_group)

        # Image info group
        info_group = QGroupBox("Image Info")
        info_layout = QVBoxLayout()
        self.info_label = QLabel("No image loaded")
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        control_layout.addWidget(info_group)

        control_layout.addStretch()

        # Apply button
        self.apply_btn = QPushButton("Apply Mask →")
        self.apply_btn.clicked.connect(self.apply_mask)
        self.apply_btn.setEnabled(False)
        control_layout.addWidget(self.apply_btn)

        splitter.addWidget(control_widget)
        splitter.setSizes([800, 300])

        layout.addWidget(splitter)

    def load_image(self):
        """Load interferogram image."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Interferogram",
            "",
            IMAGE_FILE_FILTER
        )

        if filename:
            self.image = load_image(filename)
            if self.image is not None:
                self.mask_editor.set_image(self.image)

                # Update controls
                h, w = self.image.shape[:2]
                self.center_x_spin.setValue(w // 2)
                self.center_y_spin.setValue(h // 2)
                self.center_x_spin.setMaximum(w)
                self.center_y_spin.setMaximum(h)
                self.outer_radius_spin.setMaximum(min(h, w))
                self.inner_radius_spin.setMaximum(min(h, w))
                self.outer_radius_spin.setValue(min(h, w) // 2 - 10)
                self.inner_radius_spin.setValue(20)

                # Update info
                info = get_image_info(self.image)
                self.info_label.setText(
                    f"Size: {w}x{h}\n"
                    f"Type: {info['dtype']}\n"
                    f"Min: {info['min']:.1f}\n"
                    f"Max: {info['max']:.1f}"
                )

                self.apply_btn.setEnabled(True)
                self.auto_mask_btn.setEnabled(True)
                self.reset_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Error", "Failed to load image.")

    def auto_detect_mask(self):
        """Auto-detect circular mask from image."""
        if self.image is None:
            return

        outer_params, inner_params = auto_detect_circle(self.image)

        if outer_params:
            cx, cy, r = outer_params
            self.center_x_spin.setValue(int(cx))
            self.center_y_spin.setValue(int(cy))
            self.outer_radius_spin.setValue(int(r))

        if inner_params:
            _, _, r_inner = inner_params
            self.inner_radius_spin.setValue(int(r_inner))
        else:
            # Set a small default inner radius
            self.inner_radius_spin.setValue(10)

        if not outer_params:
            QMessageBox.information(
                self,
                "Auto-detect",
                "Could not automatically detect circular features. Please adjust manually."
            )

    def reset_mask(self):
        """Reset mask to default."""
        if self.image is not None:
            h, w = self.image.shape[:2]
            self.center_x_spin.setValue(w // 2)
            self.center_y_spin.setValue(h // 2)
            self.outer_radius_spin.setValue(min(h, w) // 2 - 10)
            self.inner_radius_spin.setValue(20)

    def on_mask_changed(self, outer_r, inner_r, cx, cy):
        """Handle mask change from interactive editor."""
        # Update spinboxes without triggering their signals
        self.outer_radius_spin.blockSignals(True)
        self.inner_radius_spin.blockSignals(True)
        self.center_x_spin.blockSignals(True)
        self.center_y_spin.blockSignals(True)

        self.outer_radius_spin.setValue(int(outer_r))
        self.inner_radius_spin.setValue(int(inner_r))
        self.center_x_spin.setValue(int(cx))
        self.center_y_spin.setValue(int(cy))

        self.outer_radius_spin.blockSignals(False)
        self.inner_radius_spin.blockSignals(False)
        self.center_x_spin.blockSignals(False)
        self.center_y_spin.blockSignals(False)

    def on_outer_mask_toggled(self, checked):
        """Handle outer mask checkbox toggle."""
        self.outer_radius_spin.setEnabled(checked)
        self.update_mask_from_controls()

    def on_inner_mask_toggled(self, checked):
        """Handle inner mask checkbox toggle."""
        self.inner_radius_spin.setEnabled(checked)
        self.update_mask_from_controls()

    def update_mask_from_controls(self):
        """Update mask editor from spinbox controls."""
        # Get outer radius (use large value if outer mask disabled)
        if self.use_outer_mask_check.isChecked():
            outer_r = self.outer_radius_spin.value()
        else:
            # No outer limit - use image diagonal
            if self.image is not None:
                h, w = self.image.shape[:2]
                outer_r = int(np.sqrt(h**2 + w**2))
            else:
                outer_r = 10000  # Large default

        # Get inner radius (use 0 if inner mask disabled)
        if self.use_inner_mask_check.isChecked():
            inner_r = self.inner_radius_spin.value()
        else:
            inner_r = 0

        cx = self.center_x_spin.value()
        cy = self.center_y_spin.value()

        self.mask_editor.set_mask_parameters(outer_r, inner_r, cx, cy)

    def apply_mask(self):
        """Apply mask and proceed to editing tab."""
        if self.image is None:
            return

        # Generate mask with current settings
        # Get effective radii based on checkboxes
        if self.use_outer_mask_check.isChecked():
            outer_r = self.outer_radius_spin.value()
        else:
            # No outer limit
            h, w = self.image.shape[:2]
            outer_r = int(np.sqrt(h**2 + w**2))

        if self.use_inner_mask_check.isChecked():
            inner_r = self.inner_radius_spin.value()
        else:
            inner_r = 0

        # Generate mask with effective radii
        from core.masking import generate_circular_mask
        self.mask = generate_circular_mask(
            self.image.shape[:2],
            (self.center_x_spin.value(), self.center_y_spin.value()),
            outer_r,
            inner_r
        )

        # Store in main window
        self.main_window.current_image = self.image
        self.main_window.mask = self.mask

        # Enable editing tab and switch to it
        self.main_window.enable_tab(1)
        self.main_window.switch_to_tab(1)

        # Initialize editing tab with image
        self.main_window.editing_tab.set_image(self.image, self.mask)

        QMessageBox.information(
            self,
            "Success",
            "Mask applied successfully! Proceed to the Editing tab."
        )
