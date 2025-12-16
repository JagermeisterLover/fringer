# Interferogram Analyzer

Professional interferogram analysis application for optical engineers.

## Overview

Interferogram Analyzer is a comprehensive desktop application for analyzing interferometric measurements. It provides tools for phase extraction, unwrapping, wavefront analysis, Zernike polynomial fitting, and PSF calculation.

## Features

### Tab 1: Image Loading & Masking
- Load interferogram images (PNG, JPEG, TIFF, BMP)
- Interactive circular mask editor with drag-and-drop functionality
- Automatic mask detection using Hough Circle Transform
- Support for central obscuration (annular aperture)
- Real-time mask parameter adjustment

### Tab 2: Image Editing & Preprocessing
- **Denoising Methods:**
  - Non-local means denoising
  - Bilateral filtering (edge-preserving)
  - Median filtering
- **Smoothing Methods:**
  - Gaussian smoothing
  - Anisotropic diffusion
- **Corrections:**
  - Vignette removal
  - Contrast enhancement (CLAHE)
- Full undo/redo support (20 levels)
- Real-time preview

### Tab 3: Analysis & Results
- **Phase Extraction:** FFT-based method with automatic carrier frequency detection
- **Phase Unwrapping:** Quality-guided unwrapping with piston/tilt removal
- **Wavefront Analysis:**
  - 3D visualization using PyQtGraph OpenGL
  - RMS and Peak-to-Valley metrics
  - Wavefront quality assessment
- **PSF Calculation:**
  - Point Spread Function from wavefront data
  - Strehl ratio calculation
  - Log-scale visualization
- **Zernike Analysis:**
  - Polynomial fitting up to 15th order
  - Coefficient visualization (bar chart)
  - Term-by-term breakdown
  - Reconstruction and residual analysis

## Installation

### Requirements
- Python 3.10 or higher
- PyQt6
- NumPy
- OpenCV
- scikit-image
- SciPy
- Matplotlib
- PyQtGraph

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fringer.git
cd fringer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
python main.py
```

### Workflow

1. **Load and Mask (Tab 1)**
   - Click "Load Image" to open an interferogram
   - Adjust the circular mask by dragging the circles or using spinboxes
   - Use "Auto-detect Mask" for automatic detection
   - Click "Apply Mask" to proceed

2. **Edit and Process (Tab 2)**
   - Apply denoising to reduce noise
   - Use smoothing for better phase extraction
   - Remove vignetting if present
   - Click "Calculate Phase" when ready

3. **Analyze Results (Tab 3)**
   - Click "Run Phase Analysis" to extract and unwrap phase
   - Explore different sub-tabs:
     - **Phase Maps:** View wrapped and unwrapped phase
     - **3D Wavefront:** Interactive 3D visualization
     - **PSF:** Point Spread Function and Strehl ratio
     - **Zernike Analysis:** Fit polynomials and view coefficients

## Project Structure

```
fringer/
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── config/
│   └── settings.py         # Application settings and constants
│
├── core/
│   ├── image_loader.py     # Image loading and validation
│   ├── masking.py          # Mask generation and operations
│   ├── preprocessing.py    # Denoising and smoothing
│   ├── phase_extraction.py # FFT-based phase extraction
│   ├── phase_unwrapping.py # Phase unwrapping algorithms
│   ├── zernike.py          # Zernike polynomials
│   └── psf_calculator.py   # PSF calculation
│
├── gui/
│   ├── main_window.py      # Main application window
│   ├── tabs/
│   │   ├── loading_tab.py  # Tab 1: Loading & Masking
│   │   ├── editing_tab.py  # Tab 2: Editing & Preprocessing
│   │   └── analysis_tab.py # Tab 3: Analysis & Results
│   └── widgets/
│       ├── mask_editor.py       # Interactive mask editor
│       └── wavefront_3d_viewer.py # 3D visualization
│
├── utils/
│   └── history_manager.py  # Undo/redo functionality
│
└── algorithms/
    └── metrics.py          # Wavefront quality metrics
```

## Technical Details

### Phase Extraction Algorithm

Uses Fourier Transform method:
1. Compute 2D FFT of interferogram
2. Identify carrier frequency (highest peak in FFT)
3. Apply Gaussian bandpass filter centered at carrier frequency
4. Inverse FFT to obtain complex field
5. Extract phase using arctan2

### Phase Unwrapping

Quality-guided unwrapping using scikit-image with:
- Phase derivative variance as quality metric
- Automatic piston and tilt removal
- NaN handling for masked regions

### Zernike Polynomials

- Noll indexing convention
- Least-squares fitting
- Support for term exclusion
- Reconstruction and residual analysis

## Troubleshooting

### Common Issues

**Issue: Application doesn't start**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.10+)

**Issue: Image loading fails**
- Verify image format is supported (PNG, JPEG, TIFF, BMP)
- Check image is grayscale or will be converted automatically

**Issue: Phase extraction gives poor results**
- Try preprocessing first (denoising, smoothing)
- Ensure mask covers the interferogram aperture correctly
- Check for sufficient fringe visibility

**Issue: 3D visualization is slow**
- Large images are automatically downsampled for performance
- Close other resource-intensive applications

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Your License Here]

## Acknowledgments

- Based on standard interferometry analysis techniques
- Uses PyQt6 for GUI
- Scientific computing with NumPy, SciPy, and scikit-image
- Visualization with Matplotlib and PyQtGraph

## Contact

For questions or support, please open an issue on GitHub.

## Citation

If you use this software in your research, please cite:

```
[Your Citation Information]
```
