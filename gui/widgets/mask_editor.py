"""
Interactive circular mask editor widget.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QPainter, QImage, QPen, QColor, QPixmap
from core.masking import generate_circular_mask
from config.settings import (
    MASK_OVERLAY_ALPHA,
    MASK_COLOR_OUTER,
    MASK_COLOR_INNER,
    CIRCLE_EDGE_WIDTH,
    HOVER_CIRCLE_WIDTH
)


class MaskEditor(QWidget):
    """
    Interactive circular mask editor with draggable circles.

    Features:
    - Display image as background
    - Overlay two circles (outer/inner)
    - Draggable circles (click and drag edge)
    - Emit signals when mask parameters change
    - Synchronize with spinbox controls
    """

    maskChanged = pyqtSignal(float, float, float, float)  # outer_r, inner_r, center_x, center_y

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)

        # State variables
        self.image = None
        self.q_image = None
        self.outer_radius = 100
        self.inner_radius = 30
        self.center_x = 200
        self.center_y = 200
        self.dragging_element = None  # 'outer', 'inner', 'center_outer', 'center_inner'
        self.hover_element = None
        self.mask_visible = True
        self.drag_offset = QPoint(0, 0)

        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)

    def set_image(self, image: np.ndarray):
        """Set the background image."""
        self.image = image
        h, w = image.shape[:2]

        # Convert to QImage
        if len(image.shape) == 2:
            # Grayscale
            q_img = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            # Color
            q_img = QImage(image.data, w, h, w * 3, QImage.Format.Format_RGB888)

        self.q_image = q_img.copy()

        # Initialize mask parameters
        self.center_x = w / 2
        self.center_y = h / 2
        self.outer_radius = min(h, w) / 2 - 20
        self.inner_radius = 20

        self.update()

    def set_mask_parameters(self, outer_r: float, inner_r: float, cx: float, cy: float):
        """Set mask parameters programmatically."""
        self.outer_radius = outer_r
        self.inner_radius = inner_r
        self.center_x = cx
        self.center_y = cy
        self.update()

    def set_mask_visible(self, visible: bool):
        """Set mask visibility."""
        self.mask_visible = visible
        self.update()

    def get_display_transform(self):
        """Calculate transformation from image coordinates to widget coordinates."""
        if self.q_image is None or self.image is None:
            return 1.0, 0, 0

        # Get original image size
        img_h, img_w = self.image.shape[:2]

        # Calculate scaled pixmap size
        pixmap = QPixmap.fromImage(self.q_image)
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Calculate scale factor
        scale = scaled_pixmap.width() / img_w

        # Calculate offset (centering)
        offset_x = (self.width() - scaled_pixmap.width()) // 2
        offset_y = (self.height() - scaled_pixmap.height()) // 2

        return scale, offset_x, offset_y

    def image_to_widget(self, x, y):
        """Convert image coordinates to widget coordinates."""
        scale, offset_x, offset_y = self.get_display_transform()
        return x * scale + offset_x, y * scale + offset_y

    def widget_to_image(self, x, y):
        """Convert widget coordinates to image coordinates."""
        scale, offset_x, offset_y = self.get_display_transform()
        if scale == 0:
            return x, y
        return (x - offset_x) / scale, (y - offset_y) / scale

    def paintEvent(self, event):
        """Paint the widget."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background image scaled to fit while maintaining aspect ratio
        if self.q_image is not None:
            pixmap = QPixmap.fromImage(self.q_image)
            scaled_pixmap = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            # Center the image
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)

        if self.mask_visible and self.image is not None:
            scale, offset_x, offset_y = self.get_display_transform()

            # Convert image coordinates to widget coordinates
            center_x_widget, center_y_widget = self.image_to_widget(self.center_x, self.center_y)
            outer_r_widget = self.outer_radius * scale
            inner_r_widget = self.inner_radius * scale

            # Draw outer circle
            pen_outer = QPen(QColor(*MASK_COLOR_OUTER))
            if self.hover_element == 'outer':
                pen_outer.setWidth(HOVER_CIRCLE_WIDTH)
            else:
                pen_outer.setWidth(CIRCLE_EDGE_WIDTH)
            painter.setPen(pen_outer)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                int(center_x_widget - outer_r_widget),
                int(center_y_widget - outer_r_widget),
                int(outer_r_widget * 2),
                int(outer_r_widget * 2)
            )

            # Draw inner circle
            if self.inner_radius > 0:
                pen_inner = QPen(QColor(*MASK_COLOR_INNER))
                if self.hover_element == 'inner':
                    pen_inner.setWidth(HOVER_CIRCLE_WIDTH)
                else:
                    pen_inner.setWidth(CIRCLE_EDGE_WIDTH)
                painter.setPen(pen_inner)
                painter.drawEllipse(
                    int(center_x_widget - inner_r_widget),
                    int(center_y_widget - inner_r_widget),
                    int(inner_r_widget * 2),
                    int(inner_r_widget * 2)
                )

            # Draw center crosshair
            pen_center = QPen(QColor(255, 255, 0))
            pen_center.setWidth(2)
            painter.setPen(pen_center)
            cross_size = 10
            painter.drawLine(
                int(center_x_widget - cross_size), int(center_y_widget),
                int(center_x_widget + cross_size), int(center_y_widget)
            )
            painter.drawLine(
                int(center_x_widget), int(center_y_widget - cross_size),
                int(center_x_widget), int(center_y_widget + cross_size)
            )

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            x_widget, y_widget = pos.x(), pos.y()

            # Convert widget coordinates to image coordinates
            x_img, y_img = self.widget_to_image(x_widget, y_widget)

            # Check which element is clicked (in image coordinates)
            distance_from_center = np.sqrt((x_img - self.center_x)**2 + (y_img - self.center_y)**2)

            # Check if clicking on center
            if distance_from_center < 15:  # 15 pixel tolerance
                self.dragging_element = 'center'
                self.drag_offset = QPoint(int(self.center_x - x_img), int(self.center_y - y_img))
            # Check if clicking on outer circle edge
            elif abs(distance_from_center - self.outer_radius) < 10:  # 10 pixel tolerance
                self.dragging_element = 'outer'
            # Check if clicking on inner circle edge
            elif abs(distance_from_center - self.inner_radius) < 10 and self.inner_radius > 0:
                self.dragging_element = 'inner'

    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        pos = event.pos()
        x_widget, y_widget = pos.x(), pos.y()

        # Convert widget coordinates to image coordinates
        x_img, y_img = self.widget_to_image(x_widget, y_widget)

        if self.dragging_element is None:
            # Update hover state (in image coordinates)
            distance_from_center = np.sqrt((x_img - self.center_x)**2 + (y_img - self.center_y)**2)

            if abs(distance_from_center - self.outer_radius) < 10:
                self.hover_element = 'outer'
            elif abs(distance_from_center - self.inner_radius) < 10 and self.inner_radius > 0:
                self.hover_element = 'inner'
            else:
                self.hover_element = None

            self.update()
        else:
            # Handle dragging (in image coordinates)
            if self.dragging_element == 'center':
                # Move center
                self.center_x = x_img + self.drag_offset.x()
                self.center_y = y_img + self.drag_offset.y()

                # Clamp to image bounds
                if self.image is not None:
                    h, w = self.image.shape[:2]
                    self.center_x = max(0, min(w, self.center_x))
                    self.center_y = max(0, min(h, self.center_y))

            elif self.dragging_element == 'outer':
                # Resize outer circle
                distance = np.sqrt((x_img - self.center_x)**2 + (y_img - self.center_y)**2)
                self.outer_radius = max(self.inner_radius + 10, distance)
            elif self.dragging_element == 'inner':
                # Resize inner circle
                distance = np.sqrt((x_img - self.center_x)**2 + (y_img - self.center_y)**2)
                self.inner_radius = max(0, min(distance, self.outer_radius - 10))

            # Emit signal
            self.maskChanged.emit(self.outer_radius, self.inner_radius, self.center_x, self.center_y)
            self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging_element = None
            self.update()

    def generate_mask(self) -> np.ndarray:
        """Generate binary mask array based on current parameters."""
        if self.image is None:
            return None

        mask = generate_circular_mask(
            self.image.shape[:2],
            (self.center_x, self.center_y),
            self.outer_radius,
            self.inner_radius
        )

        return mask
