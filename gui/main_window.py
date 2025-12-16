"""
Main application window for Interferogram Analyzer.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QVBoxLayout, QWidget,
    QMenuBar, QToolBar, QMessageBox
)
from PyQt6.QtCore import Qt
from config.settings import APP_NAME, APP_VERSION, WINDOW_WIDTH, WINDOW_HEIGHT, DEFAULT_WAVELENGTH


class MainWindow(QMainWindow):
    """Main application window with tab structure."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Shared data between tabs
        self.current_image = None
        self.mask = None
        self.processed_image = None
        self.wavelength = DEFAULT_WAVELENGTH

        # Phase analysis results
        self.wrapped_phase = None
        self.unwrapped_phase = None
        self.wavefront = None
        self.zernike_coefficients = None

        self.setup_ui()
        self.create_menu_bar()
        self.create_toolbar()

    def setup_ui(self):
        """Setup main UI with tabs."""
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tabs = QTabWidget()

        # Import tabs here to avoid circular imports
        from gui.tabs.loading_tab import LoadingTab
        from gui.tabs.editing_tab import EditingTab
        from gui.tabs.analysis_tab import AnalysisTab

        # Create tabs
        self.loading_tab = LoadingTab(self)
        self.editing_tab = EditingTab(self)
        self.analysis_tab = AnalysisTab(self)

        # Add tabs
        self.tabs.addTab(self.loading_tab, "1. Load & Mask")
        self.tabs.addTab(self.editing_tab, "2. Edit & Process")
        self.tabs.addTab(self.analysis_tab, "3. Analysis")

        # Disable tabs 2 and 3 initially
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)

        layout.addWidget(self.tabs)
        self.setCentralWidget(central_widget)

    def create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open Interferogram...", self.loading_tab.load_image)
        file_menu.addAction("Save Project...", self.save_project)
        file_menu.addAction("Export Results...", self.export_results)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Undo", self.editing_tab.undo)
        edit_menu.addAction("Redo", self.editing_tab.redo)

        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.show_about)

    def create_toolbar(self):
        """Create toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        toolbar.addAction("Open", self.loading_tab.load_image)
        toolbar.addAction("Save", self.save_project)
        toolbar.addSeparator()
        toolbar.addAction("Undo", self.editing_tab.undo)
        toolbar.addAction("Redo", self.editing_tab.redo)

    def enable_tab(self, tab_index: int):
        """Enable specific tab."""
        self.tabs.setTabEnabled(tab_index, True)

    def switch_to_tab(self, tab_index: int):
        """Switch to specific tab."""
        self.tabs.setCurrentIndex(tab_index)

    def save_project(self):
        """Save current project state."""
        QMessageBox.information(self, "Save Project", "Save project functionality not yet implemented.")

    def export_results(self):
        """Export analysis results."""
        QMessageBox.information(self, "Export Results", "Export results functionality not yet implemented.")

    def show_about(self):
        """Show about dialog."""
        about_text = f"""
        <h2>{APP_NAME}</h2>
        <p>Version {APP_VERSION}</p>
        <p>Professional interferogram analysis application for optical engineers.</p>
        <p>Features:</p>
        <ul>
            <li>Phase extraction using FFT</li>
            <li>Phase unwrapping</li>
            <li>Wavefront analysis</li>
            <li>Zernike polynomial fitting</li>
            <li>PSF calculation</li>
        </ul>
        """
        QMessageBox.about(self, "About", about_text)
