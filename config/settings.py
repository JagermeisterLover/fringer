"""
Application settings and constants for Interferogram Analyzer.
"""

import numpy as np

# Application Information
APP_NAME = "Interferogram Analyzer"
APP_VERSION = "1.0.0"
ORGANIZATION = "OpticsSoftware"

# Default Wavelengths (in meters)
WAVELENGTHS = {
    'HeNe (632.8nm)': 632.8e-9,
    'Nd:YAG (1064nm)': 1064e-9,
    'Green (532nm)': 532e-9,
    'Blue (488nm)': 488e-9,
}
DEFAULT_WAVELENGTH = 632.8e-9  # HeNe laser

# Image Processing Settings
MAX_IMAGE_SIZE = 4096  # Maximum image dimension
DEFAULT_MASK_INNER_RADIUS = 20
DEFAULT_MASK_OUTER_RADIUS_RATIO = 0.45  # Relative to image size

# History Settings
MAX_HISTORY_LENGTH = 20

# Phase Extraction Settings
FFT_FILTER_SIGMA = 20  # Gaussian bandpass filter bandwidth
DC_MASK_RADIUS = 5  # Radius to mask DC component in FFT

# Phase Unwrapping Settings
UNWRAP_QUALITY_THRESHOLD = 0.5

# Zernike Settings
DEFAULT_ZERNIKE_MAX_ORDER = 6
MAX_ZERNIKE_ORDER = 15
ZERNIKE_NORMALIZATION = 'noll'  # 'noll' or 'fringe'

# PSF Calculation Settings
PSF_FFT_SIZE = 2048
PSF_OVERSAMPLING = 4

# Visualization Settings
COLORMAPS = [
    'jet',
    'twilight',
    'hsv',
    'viridis',
    'plasma',
    'inferno',
    'coolwarm',
    'seismic',
]
DEFAULT_COLORMAP = 'jet'

# 3D Visualization
VIEW_DISTANCE = 50
VIEW_ELEVATION = 30
VIEW_AZIMUTH = 45
SURFACE_COLOR_LOW = (0.2, 0.2, 1.0, 1.0)  # Blue
SURFACE_COLOR_MID = (0.0, 1.0, 0.0, 1.0)  # Green
SURFACE_COLOR_HIGH = (1.0, 0.0, 0.0, 1.0)  # Red

# Export Settings
EXPORT_DPI = 300
EXPORT_IMAGE_FORMATS = ['PNG', 'TIFF', 'JPEG', 'BMP']
EXPORT_DATA_FORMATS = ['CSV', 'NPY', 'TXT']

# UI Settings
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
SPLITTER_RATIO = [800, 300]  # Left/Right split ratio
MASK_OVERLAY_ALPHA = 0.3
MASK_COLOR_OUTER = (0, 255, 0)  # Green
MASK_COLOR_INNER = (255, 0, 0)  # Red
CIRCLE_EDGE_WIDTH = 3
HOVER_CIRCLE_WIDTH = 5

# Preprocessing Default Parameters
DENOISE_NLM_H = 10
DENOISE_NLM_TEMPLATE_SIZE = 7
DENOISE_NLM_SEARCH_SIZE = 21

DENOISE_BILATERAL_D = 9
DENOISE_BILATERAL_SIGMA_COLOR = 75
DENOISE_BILATERAL_SIGMA_SPACE = 75

DENOISE_MEDIAN_KERNEL = 5

SMOOTH_GAUSSIAN_SIGMA = 1.0
SMOOTH_ANISOTROPIC_ITERATIONS = 50
SMOOTH_ANISOTROPIC_KAPPA = 50
SMOOTH_ANISOTROPIC_GAMMA = 0.1

VIGNETTE_POLY_DEGREE = 2

# Notch Filter
NOTCH_RADIUS = 10

# Metrics Units
UNIT_WAVES = 'waves'
UNIT_NANOMETERS = 'nm'
UNIT_MICROMETERS = 'μm'

# File Filters
IMAGE_FILE_FILTER = "Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp);;All Files (*)"
PROJECT_FILE_FILTER = "Interferogram Project (*.ifp);;All Files (*)"
