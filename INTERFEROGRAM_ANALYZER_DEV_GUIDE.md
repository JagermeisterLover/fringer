# Interferogram Analysis Program - Complete Development Guide

## Project Overview

This is a comprehensive guide for developing a professional interferogram analysis application using PyQt6. The program performs phase extraction, unwrapping, wavefront analysis, and Zernike polynomial fitting from interferometric measurements.

**Target User:** Optical engineers working with interferometry
**Primary Use:** Analyze interferograms to extract wavefront information, calculate optical aberrations, and perform quality assessment

---

## Technical Specifications

### Technology Stack

- **Framework:** PyQt6 (GUI)
- **Python Version:** 3.10+
- **Image Processing:** OpenCV, scikit-image
- **Scientific Computing:** NumPy, SciPy
- **Visualization:** Matplotlib (embedded), PyQtGraph (3D), Plotly (optional)
- **Phase Processing:** scikit-image (unwrapping), custom FFT algorithms

### Dependencies (requirements.txt)

```txt
PyQt6>=6.6.0
PyQt6-WebEngine>=6.6.0
numpy>=1.24.0
opencv-python>=4.8.0
scikit-image>=0.21.0
scipy>=1.11.0
matplotlib>=3.7.0
pyqtgraph>=0.13.0
pillow>=10.0.0
```

---

## Complete File Structure

```
interferogram_analyzer/
│
├── main.py                          # Application entry point
├── requirements.txt                 # Dependencies
├── README.md                        # User documentation
├── .gitignore
│
├── config/
│   ├── __init__.py
│   └── settings.py                  # Application settings, constants
│
├── gui/
│   ├── __init__.py
│   ├── main_window.py               # Main application window with tab structure
│   │
│   ├── tabs/
│   │   ├── __init__.py
│   │   ├── loading_tab.py           # Tab 1: Image loading & masking
│   │   ├── editing_tab.py           # Tab 2: Image editing & preprocessing
│   │   └── analysis_tab.py          # Tab 3: Results with nested tabs
│   │
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── mask_editor.py           # Interactive circular mask widget
│   │   ├── image_viewer.py          # Custom image display with zoom/pan
│   │   ├── phase_map_viewer.py      # Wrapped/unwrapped phase display
│   │   ├── wavefront_3d_viewer.py   # 3D wavefront visualization
│   │   ├── psf_viewer.py            # FFT PSF display
│   │   ├── zernike_panel.py         # Zernike fitting controls & results
│   │   └── parameter_panel.py       # Reusable parameter control panel
│   │
│   └── dialogs/
│       ├── __init__.py
│       ├── export_dialog.py         # Export results dialog
│       └── settings_dialog.py       # Application settings
│
├── core/
│   ├── __init__.py
│   ├── image_loader.py              # Image loading & validation
│   ├── masking.py                   # Circular mask generation & operations
│   ├── preprocessing.py             # Denoising, smoothing, vignette removal
│   ├── phase_extraction.py          # Interferogram → phase (FFT method)
│   ├── phase_unwrapping.py          # Phase unwrapping algorithms
│   ├── zernike.py                   # Zernike polynomial generation & fitting
│   ├── psf_calculator.py            # PSF calculation from wavefront
│   └── fringe_removal.py            # Parasitic fringe removal algorithms
│
├── utils/
│   ├── __init__.py
│   ├── image_operations.py          # Common image operations (normalize, crop, etc.)
│   ├── history_manager.py           # Undo/redo functionality
│   ├── validators.py                # Input validation
│   └── export.py                    # Export functionality (images, data, reports)
│
├── algorithms/
│   ├── __init__.py
│   ├── fourier_transform.py         # FFT-based phase extraction
│   ├── filtering.py                 # Various filtering methods
│   └── metrics.py                   # RMS, P-V, Strehl calculations
│
└── resources/
    ├── icons/                       # UI icons
    ├── styles/
    │   ├── dark_theme.qss           # Dark Qt stylesheet
    │   └── light_theme.qss          # Light Qt stylesheet
    └── examples/                    # Example interferograms for testing
```

---

## Detailed Feature Specifications

### TAB 1: Image Loading & Masking

#### Requirements

1. **Image Loading**
   - Support formats: PNG, JPEG, TIFF, BMP
   - Display image in custom viewer
   - Show image properties (dimensions, bit depth, file size)
   - Handle grayscale and color images (convert to grayscale if needed)

2. **Interactive Masking**
   - Display loaded image with overlay
   - Two circular masks: outer (aperture) and inner (central obscuration)
   - **Interactive controls:**
     - Drag circle edges with mouse
     - Drag circle centers to reposition
     - Visual feedback (highlighted when hovering)
   - **Manual adjustment controls:**
     - Outer radius (spinbox, 0-max)
     - Inner radius (spinbox, 0-outer_radius)
     - Center X (spinbox, 0-width)
     - Center Y (spinbox, 0-height)
     - All controls synchronized with interactive view
   
3. **Mask Visualization**
   - Semi-transparent overlay (alpha=0.3)
   - Different colors for outer/inner masks
   - Show mask as red tint on masked regions
   - Toggle mask visibility
   
4. **Controls**
   - Load Image button
   - Apply Mask button (proceeds to Tab 2)
   - Reset Mask button
   - Auto-detect mask (find circular features)
   - Save mask configuration

#### Implementation Details

**MaskEditor Widget:**
```python
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
        # State variables
        self.image = None
        self.outer_radius = 100
        self.inner_radius = 30
        self.center_x = 0
        self.center_y = 0
        self.dragging_element = None  # 'outer', 'inner', 'center'
        
    def paintEvent(self, event):
        # Draw image
        # Draw outer circle (thick line)
        # Draw inner circle (thick line)
        # Draw center crosshair
        # Highlight hovered element
        
    def mousePressEvent(self, event):
        # Detect which element is clicked (outer edge, inner edge, center)
        # Set dragging_element
        
    def mouseMoveEvent(self, event):
        # Update radius or center based on dragging_element
        # Emit maskChanged signal
        # Update display
        
    def mouseReleaseEvent(self, event):
        # Clear dragging_element
        
    def generate_mask(self):
        """Generate binary mask array based on current parameters."""
        # Create coordinate grid
        # Calculate distance from center
        # mask = (distance < outer_radius) & (distance > inner_radius)
        return mask
```

**LoadingTab Layout:**
```
┌─────────────────────────────────────────────────┐
│  [Load Image] [Auto-detect Mask] [Reset]        │
├─────────────────────────┬───────────────────────┤
│                         │ Mask Parameters       │
│                         │                       │
│   Image Display Area    │ Outer Radius: [___]  │
│   (MaskEditor Widget)   │ Inner Radius: [___]  │
│                         │ Center X: [___]      │
│                         │ Center Y: [___]      │
│                         │                       │
│                         │ [x] Show Mask         │
│                         │                       │
│                         │ Image Info:          │
│                         │ Size: 1024x1024      │
│                         │ Bit Depth: 8-bit     │
│                         │                       │
│                         │ [Apply Mask →]       │
└─────────────────────────┴───────────────────────┘
```

---

### TAB 2: Image Editing & Preprocessing

#### Requirements

1. **Denoising Filters**
   - **Non-local means denoising**
     - Parameters: h (filter strength), template window size, search window size
     - Method: `cv2.fastNlMeansDenoising()`
   - **Bilateral filter**
     - Parameters: d (diameter), sigma_color, sigma_space
     - Edge-preserving smoothing
   - **Median filter**
     - Parameter: kernel size (3, 5, 7, 9)
     - Good for salt-and-pepper noise

2. **Smoothing**
   - **Gaussian smoothing**
     - Parameter: sigma (0.5 - 5.0)
     - Method: `cv2.GaussianBlur()`
   - **Anisotropic diffusion**
     - Parameters: iterations, kappa, gamma
     - Edge-preserving while smoothing
     - Method: Custom implementation or `skimage.filters.denoise_tv_chambolle()`

3. **Vignette Removal**
   - **Automatic correction:**
     - Fit 2D polynomial to intensity distribution
     - Divide image by fitted surface
   - **Manual adjustment:**
     - Strength slider
     - Center offset adjustments

4. **Fringe Removal**
   - **Fourier domain notch filter:**
     - Display FFT magnitude spectrum
     - User selects frequency peaks to suppress
     - Apply notch filter and inverse FFT
   - **High-pass filter:**
     - Remove low-frequency background
   - **Median filter variant:**
     - Specialized for periodic noise

5. **Undo/Redo System**
   - Track all operations in history
   - Maximum 20 operations
   - Undo/Redo buttons always available
   - Show operation history list
   - Memory-efficient (store only differences if possible)

6. **UI Features**
   - **Before/After Split View**
     - Draggable divider
     - Shows original on left, processed on right
   - **Parameter Controls**
     - Sliders with numeric input
     - Preview checkbox (apply in real-time vs on button click)
     - Apply/Cancel for each operation
   - **Operation History Panel**
     - List of applied operations
     - Click to jump to that state
     - Delete operation from history

#### Implementation Details

**History Manager:**
```python
class HistoryManager:
    """
    Manages undo/redo for image editing operations.
    
    Stores image states and operation descriptions.
    """
    
    def __init__(self, max_history=20):
        self.history = []  # List of (image, description) tuples
        self.current_index = -1
        self.max_history = max_history
        
    def add_state(self, image, description):
        """Add new state, remove any future states."""
        # Remove states after current_index
        self.history = self.history[:self.current_index + 1]
        
        # Add new state
        self.history.append((image.copy(), description))
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.current_index += 1
            
    def undo(self):
        """Go back one state."""
        if self.can_undo():
            self.current_index -= 1
            return self.history[self.current_index][0].copy()
        return None
        
    def redo(self):
        """Go forward one state."""
        if self.can_redo():
            self.current_index += 1
            return self.history[self.current_index][0].copy()
        return None
        
    def can_undo(self):
        return self.current_index > 0
        
    def can_redo(self):
        return self.current_index < len(self.history) - 1
        
    def get_current(self):
        """Get current image state."""
        if self.history:
            return self.history[self.current_index][0].copy()
        return None
        
    def get_history_list(self):
        """Get list of operation descriptions."""
        return [desc for _, desc in self.history[:self.current_index + 1]]
        
    def reset(self):
        """Clear history."""
        self.history = []
        self.current_index = -1
```

**Preprocessing Functions:**
```python
# core/preprocessing.py

def denoise_nonlocal_means(image, h=10, template_window_size=7, search_window_size=21):
    """Apply non-local means denoising."""
    return cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)

def denoise_bilateral(image, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filtering (edge-preserving)."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def smooth_gaussian(image, sigma=1.0):
    """Apply Gaussian smoothing."""
    return cv2.GaussianBlur(image, (0, 0), sigma)

def smooth_anisotropic(image, iterations=50, kappa=50, gamma=0.1):
    """
    Apply anisotropic diffusion (edge-preserving smoothing).
    Uses Perona-Malik algorithm.
    """
    from skimage.restoration import denoise_tv_chambolle
    # Alternative: implement custom Perona-Malik
    return denoise_tv_chambolle(image, weight=0.1, max_num_iter=iterations)

def remove_vignette(image, polynomial_degree=2):
    """
    Remove vignetting by fitting and dividing by polynomial surface.
    """
    h, w = image.shape
    y, x = np.ogrid[:h, :w]
    
    # Normalize coordinates to [-1, 1]
    x = (x - w/2) / (w/2)
    y = (y - h/2) / (h/2)
    
    # Create polynomial terms
    # For degree 2: 1, x, y, x^2, xy, y^2
    terms = []
    for i in range(polynomial_degree + 1):
        for j in range(polynomial_degree + 1 - i):
            terms.append((x**j) * (y**i))
    
    # Stack terms into design matrix
    A = np.stack(terms, axis=-1).reshape(-1, len(terms))
    b = image.flatten()
    
    # Least squares fit
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Reconstruct fitted surface
    fitted = A @ coeffs
    fitted = fitted.reshape(h, w)
    
    # Normalize to prevent divide-by-zero
    fitted = np.maximum(fitted, np.percentile(fitted, 5))
    
    # Correct vignetting
    corrected = image / fitted
    corrected = (corrected / corrected.max() * 255).astype(np.uint8)
    
    return corrected

def remove_fringes_fft_notch(image, notch_frequencies):
    """
    Remove parasitic fringes by notch filtering in Fourier domain.
    
    Args:
        image: Input image
        notch_frequencies: List of (u, v) frequency coordinates to suppress
    
    Returns:
        Filtered image
    """
    # FFT
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    
    # Create notch filter
    h, w = image.shape
    notch_filter = np.ones((h, w))
    
    for u, v in notch_frequencies:
        # Create Gaussian notch centered at (u, v)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # Notch at (u, v) and (-u, -v) for symmetry
        radius = 10  # Notch width
        mask1 = np.exp(-((x - (center_x + u))**2 + (y - (center_y + v))**2) / (2 * radius**2))
        mask2 = np.exp(-((x - (center_x - u))**2 + (y - (center_y - v))**2) / (2 * radius**2))
        notch_filter *= (1 - mask1) * (1 - mask2)
    
    # Apply filter
    filtered_fft = fft_shifted * notch_filter
    
    # Inverse FFT
    filtered = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
    filtered = np.abs(filtered).astype(np.uint8)
    
    return filtered
```

**EditingTab Layout:**
```
┌────────────────────────────────────────────────────────────┐
│ [Undo] [Redo] [Reset to Original]    History: [▼]         │
├─────────────────────────┬──────────────────────────────────┤
│                         │                                  │
│  Before | After         │  Denoising                       │
│  (Split View)           │  ○ Non-local Means               │
│                         │    h: [━━●━━━━━] 10              │
│                         │  ○ Bilateral Filter              │
│                         │    d: [━●━━━━━━] 9               │
│                         │  ○ Median Filter                 │
│                         │    kernel: [5▼]                  │
│                         │  [Apply]                         │
│                         │                                  │
│                         │  Smoothing                       │
│                         │  ○ Gaussian                      │
│                         │    sigma: [━━●━━━] 1.5           │
│                         │  ○ Anisotropic                   │
│                         │    iterations: [━━━●━━] 50       │
│                         │  [Apply]                         │
│                         │                                  │
│                         │  Corrections                     │
│                         │  □ Remove Vignette               │
│                         │  □ Remove Fringes (FFT)          │
│                         │  [Apply]                         │
│                         │                                  │
│                         │  [Preview] [Calculate →]         │
└─────────────────────────┴──────────────────────────────────┘
```

---

### TAB 3: Analysis & Results (Nested Tabs)

#### Requirements Overview

Tab 3 contains nested sub-tabs:
1. **Wrapped Phase** - Display phase map (-π to π)
2. **Unwrapped Phase** - Display continuous phase
3. **3D Wavefront** - 3D surface plot with metrics
4. **FFT PSF** - Point spread function
5. **Zernike Analysis** - Polynomial fitting and coefficients

All tabs should have:
- Export button (save image/data)
- Colormap selector
- Zoom/pan tools
- Measurement tools

---

#### Sub-Tab 1: Wrapped Phase Map

**Features:**
- Display phase map with values in range [-π, π]
- Color scale showing phase values
- Colormap selection (jet, twilight, hsv, phase colormap)
- Statistics panel:
  - Min/Max phase
  - Mean, Std deviation
  - Phase discontinuities count
- Export as image (PNG, TIFF)

**Phase Extraction Algorithm (FFT Method):**

```python
# core/phase_extraction.py

def extract_phase_fft(interferogram, mask=None, carrier_frequency=None):
    """
    Extract phase from interferogram using Fourier transform method.
    
    Args:
        interferogram: Input interferogram image (2D array)
        mask: Binary mask (optional)
        carrier_frequency: (fx, fy) carrier frequency in pixels (optional, auto-detect if None)
    
    Returns:
        wrapped_phase: Phase map in range [-π, π]
        fft_spectrum: FFT spectrum for visualization
    """
    
    # Apply mask if provided
    if mask is not None:
        interferogram = interferogram * mask
    
    # 1. Compute FFT
    fft = np.fft.fft2(interferogram)
    fft_shifted = np.fft.fftshift(fft)
    
    # 2. Find carrier frequency (highest peak excluding DC)
    if carrier_frequency is None:
        magnitude = np.abs(fft_shifted)
        h, w = magnitude.shape
        
        # Mask DC component
        center_y, center_x = h // 2, w // 2
        magnitude[center_y-5:center_y+5, center_x-5:center_x+5] = 0
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        carrier_frequency = (peak_idx[1] - center_x, peak_idx[0] - center_y)
    
    # 3. Create bandpass filter centered at carrier frequency
    h, w = interferogram.shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    
    # Gaussian bandpass filter
    fx, fy = carrier_frequency
    sigma = 20  # Filter bandwidth
    bandpass = np.exp(-((x - (center_x + fx))**2 + (y - (center_y + fy))**2) / (2 * sigma**2))
    
    # 4. Apply filter
    filtered_fft = fft_shifted * bandpass
    
    # 5. Shift back and inverse FFT
    filtered_fft = np.fft.ifftshift(filtered_fft)
    complex_field = np.fft.ifft2(filtered_fft)
    
    # 6. Extract phase
    wrapped_phase = np.arctan2(complex_field.imag, complex_field.real)
    
    # Apply mask to phase
    if mask is not None:
        wrapped_phase = wrapped_phase * mask
    
    return wrapped_phase, fft_shifted
```

---

#### Sub-Tab 2: Unwrapped Phase Map

**Features:**
- Display continuous (unwrapped) phase
- Phase unwrapping quality map
- Option to remove piston/tilt
- Statistics:
  - Total phase excursion
  - RMS phase
  - Peak-to-Valley (P-V)
- Export unwrapped phase as data (CSV, NPY)

**Phase Unwrapping Algorithm:**

```python
# core/phase_unwrapping.py

from skimage.restoration import unwrap_phase

def unwrap_phase_quality_guided(wrapped_phase, mask=None):
    """
    Unwrap phase using quality-guided algorithm.
    
    Args:
        wrapped_phase: Wrapped phase in range [-π, π]
        mask: Binary mask defining valid region
    
    Returns:
        unwrapped_phase: Continuous phase
        quality_map: Quality metric for each pixel
    """
    
    # Calculate quality map (phase gradient variance)
    quality_map = calculate_phase_quality(wrapped_phase)
    
    # Unwrap using scikit-image
    if mask is not None:
        # Create wrapped phase array with invalid regions set to 0
        wrapped_masked = np.copy(wrapped_phase)
        wrapped_masked[~mask] = 0
        unwrapped = unwrap_phase(wrapped_masked)
        unwrapped[~mask] = np.nan
    else:
        unwrapped = unwrap_phase(wrapped_phase)
    
    return unwrapped, quality_map

def calculate_phase_quality(wrapped_phase):
    """
    Calculate quality map based on phase derivative variance.
    High quality = low variance in local neighborhood.
    """
    # Calculate phase derivatives
    dy, dx = np.gradient(wrapped_phase)
    
    # Wrap derivatives to [-π, π]
    dy = np.arctan2(np.sin(dy), np.cos(dy))
    dx = np.arctan2(np.sin(dx), np.cos(dx))
    
    # Calculate second derivatives (variance indicator)
    d2y = np.gradient(dy)[0]
    d2x = np.gradient(dx)[1]
    
    # Quality metric (lower variance = higher quality)
    quality = 1.0 / (1.0 + d2x**2 + d2y**2)
    
    return quality

def remove_piston_tilt(phase, mask=None):
    """
    Remove piston (average) and tilt (linear trend) from phase.
    
    Args:
        phase: Input phase map
        mask: Binary mask (optional)
    
    Returns:
        phase_corrected: Phase with piston and tilt removed
    """
    h, w = phase.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    if mask is not None:
        valid = mask.astype(bool)
        x_valid = x[valid]
        y_valid = y[valid]
        z_valid = phase[valid]
    else:
        x_valid = x.flatten()
        y_valid = y.flatten()
        z_valid = phase.flatten()
    
    # Fit plane: z = a*x + b*y + c
    A = np.column_stack([x_valid, y_valid, np.ones_like(x_valid)])
    coeffs, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
    
    # Reconstruct plane
    plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    
    # Subtract plane
    phase_corrected = phase - plane
    
    if mask is not None:
        phase_corrected[~mask] = 0
    
    return phase_corrected
```

---

#### Sub-Tab 3: 3D Wavefront Map

**Features:**
- 3D surface plot of wavefront
- Interactive rotation, zoom, pan
- Colormap based on height
- **Metrics Display:**
  - RMS wavefront error (in waves or nm)
  - Peak-to-Valley (P-V)
  - Mean, Std deviation
  - Strehl ratio (if PSF calculated)
- Lighting options (surface/wireframe/points)
- Export 3D view as image
- Export wavefront data (CSV, NPY)

**Wavefront Calculation:**

```python
# Convert phase to wavefront (optical path difference)
# wavefront = (wavelength / (2 * π)) * phase

def phase_to_wavefront(unwrapped_phase, wavelength=632.8e-9):
    """
    Convert phase (radians) to wavefront (meters).
    
    Args:
        unwrapped_phase: Unwrapped phase in radians
        wavelength: Wavelength in meters (default: 632.8 nm HeNe)
    
    Returns:
        wavefront: Optical path difference in meters
    """
    wavefront = (wavelength / (2 * np.pi)) * unwrapped_phase
    return wavefront

def calculate_wavefront_metrics(wavefront, mask=None):
    """
    Calculate wavefront quality metrics.
    
    Args:
        wavefront: Wavefront map in meters
        mask: Binary mask
    
    Returns:
        metrics: Dictionary of metrics
    """
    if mask is not None:
        valid_data = wavefront[mask.astype(bool)]
    else:
        valid_data = wavefront.flatten()
    
    # Remove piston (mean)
    valid_data = valid_data - np.mean(valid_data)
    
    metrics = {
        'rms': np.sqrt(np.mean(valid_data**2)),  # RMS
        'pv': np.max(valid_data) - np.min(valid_data),  # Peak-to-Valley
        'mean': np.mean(valid_data),
        'std': np.std(valid_data),
        'min': np.min(valid_data),
        'max': np.max(valid_data)
    }
    
    return metrics
```

**3D Visualization (PyQtGraph):**

```python
# gui/widgets/wavefront_3d_viewer.py

import pyqtgraph.opengl as gl

class Wavefront3DViewer(QWidget):
    """3D viewer for wavefront data using PyQtGraph OpenGL."""
    
    def __init__(self):
        super().__init__()
        self.view = gl.GLViewWidget()
        self.surface_plot = None
        
        # Setup layout
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)
        
        # Configure view
        self.view.setCameraPosition(distance=50, elevation=30, azimuth=45)
        self.view.opts['bgcolor'] = (20, 20, 20)
        
    def display_wavefront(self, wavefront, mask=None):
        """Display wavefront as 3D surface."""
        
        # Clear existing plots
        if self.surface_plot is not None:
            self.view.removeItem(self.surface_plot)
        
        # Apply mask
        if mask is not None:
            wavefront_display = np.copy(wavefront)
            wavefront_display[~mask.astype(bool)] = np.nan
        else:
            wavefront_display = wavefront
        
        # Create surface plot
        self.surface_plot = gl.GLSurfacePlotItem(
            z=wavefront_display,
            shader='heightColor',
            computeNormals=True,
            smooth=True
        )
        
        # Color map based on height
        self.surface_plot.shader()['colorMap'] = np.array([
            [0.2, 0.2, 1.0, 1.0],  # Blue (low)
            [0.0, 1.0, 0.0, 1.0],  # Green (mid)
            [1.0, 0.0, 0.0, 1.0]   # Red (high)
        ])
        
        self.view.addItem(self.surface_plot)
```

---

#### Sub-Tab 4: FFT PSF

**Features:**
- Display Point Spread Function (PSF) calculated from wavefront
- Log scale intensity display
- Cross-section plots (horizontal/vertical)
- Encircled energy plot
- Strehl ratio calculation
- PSF metrics:
  - FWHM (Full Width Half Maximum)
  - Airy disk comparison
  - Encircled energy diameter
- Export PSF image and data

**PSF Calculation:**

```python
# core/psf_calculator.py

def calculate_psf(wavefront, wavelength=632.8e-9, pupil_diameter=1.0, pixel_size=None, fft_size=2048):
    """
    Calculate Point Spread Function from wavefront.
    
    Args:
        wavefront: Wavefront map in meters
        wavelength: Wavelength in meters
        pupil_diameter: Pupil diameter in meters
        pixel_size: Pixel size in pupil plane (meters)
        fft_size: FFT size for oversampling
    
    Returns:
        psf: Point Spread Function (normalized intensity)
        psf_image_scale: Spatial scale of PSF in micrometers/pixel
    """
    
    h, w = wavefront.shape
    
    # Create complex pupil function
    # P(x,y) = A(x,y) * exp(i * k * W(x,y))
    # where k = 2π/λ
    k = 2 * np.pi / wavelength
    pupil_function = np.exp(1j * k * wavefront)
    
    # Zero-pad for oversampling
    pad_h = (fft_size - h) // 2
    pad_w = (fft_size - w) // 2
    pupil_padded = np.pad(pupil_function, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # FFT to get PSF
    psf_complex = np.fft.fft2(pupil_padded)
    psf_complex = np.fft.fftshift(psf_complex)
    
    # Calculate intensity
    psf = np.abs(psf_complex)**2
    
    # Normalize to peak = 1
    psf = psf / np.max(psf)
    
    # Calculate spatial scale (µm/pixel in image plane)
    if pixel_size is None:
        pixel_size = pupil_diameter / h  # Approximate
    
    # Scale factor from Fourier optics
    psf_image_scale = wavelength / (pixel_size * fft_size) * 1e6  # in micrometers
    
    return psf, psf_image_scale

def calculate_strehl_ratio(psf, wavelength=632.8e-9):
    """
    Calculate Strehl ratio (peak intensity / diffraction-limited peak).
    
    Args:
        psf: Point Spread Function (normalized)
        wavelength: Wavelength in meters
    
    Returns:
        strehl: Strehl ratio (0 to 1)
    """
    # Peak of actual PSF (already normalized to 1)
    peak_actual = np.max(psf)
    
    # For normalized PSF, Strehl is approximately the peak value
    # More accurate: compare to diffraction-limited PSF
    strehl = peak_actual
    
    return strehl

def calculate_encircled_energy(psf, psf_image_scale):
    """
    Calculate encircled energy as function of radius.
    
    Args:
        psf: PSF array
        psf_image_scale: Spatial scale (µm/pixel)
    
    Returns:
        radii: Radii in micrometers
        encircled_energy: Cumulative energy fraction
    """
    h, w = psf.shape
    center_y, center_x = h // 2, w // 2
    
    # Create distance map from center
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Total energy
    total_energy = np.sum(psf)
    
    # Calculate encircled energy for each radius
    max_radius = min(center_x, center_y)
    radii = np.arange(0, max_radius)
    encircled = np.zeros_like(radii, dtype=float)
    
    for i, r in enumerate(radii):
        mask = distance <= r
        encircled[i] = np.sum(psf[mask]) / total_energy
    
    # Convert radii to micrometers
    radii_um = radii * psf_image_scale
    
    return radii_um, encircled
```

---

#### Sub-Tab 5: Zernike Analysis

**Features:**
- Fit Zernike polynomials to wavefront
- Display coefficients in bar chart
- **Term exclusion functionality:**
  - Checkboxes for each Zernike term
  - Common presets: "Exclude Piston", "Exclude Piston+Tilt", "Exclude Defocus"
  - Refit button after changing exclusions
- Reconstructed wavefront from Zernike fit
- Residual map (original - reconstructed)
- Residual RMS
- Export:
  - Zernike coefficients (CSV, TXT)
  - Bar chart image
  - Reconstructed wavefront

**Zernike Polynomial Implementation:**

```python
# core/zernike.py

import numpy as np
from scipy.special import factorial

def zernike_polynomial(n, m, rho, theta):
    """
    Calculate Zernike polynomial Z_n^m(rho, theta).
    
    Args:
        n: Radial order (non-negative integer)
        m: Azimuthal frequency (integer, |m| <= n, n-|m| even)
        rho: Normalized radial coordinate (0 to 1)
        theta: Azimuthal angle (radians)
    
    Returns:
        Z: Zernike polynomial value
    """
    if (n - abs(m)) % 2 != 0:
        return np.zeros_like(rho)
    
    # Radial polynomial R_n^m(rho)
    R = radial_polynomial(n, abs(m), rho)
    
    # Azimuthal component
    if m >= 0:
        Z = R * np.cos(m * theta)
    else:
        Z = R * np.sin(abs(m) * theta)
    
    return Z

def radial_polynomial(n, m, rho):
    """Calculate radial polynomial R_n^m(rho)."""
    R = np.zeros_like(rho, dtype=float)
    
    for k in range((n - m) // 2 + 1):
        num = (-1)**k * factorial(n - k)
        den = factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        R += (num / den) * rho**(n - 2*k)
    
    return R

def noll_to_nm(j):
    """
    Convert Noll index j to (n, m) Zernike indices.
    
    Noll ordering is standard in optics (starts at j=1 for piston).
    """
    n = 0
    while j > n + 1:
        n += 1
        j -= n
    
    m = n - 2 * (j - 1)
    if n % 2 == 0:
        m = -m if j % 2 == 0 else m
    else:
        m = m if j % 2 == 0 else -m
    
    return n, m

def generate_zernike_basis(size, max_order=6, mask=None):
    """
    Generate Zernike basis functions up to specified order.
    
    Args:
        size: Image size (assumes square, size x size)
        max_order: Maximum radial order n
        mask: Binary circular mask
    
    Returns:
        basis: List of Zernike polynomials (2D arrays)
        indices: List of (j, n, m) tuples for each basis function
    """
    # Create normalized coordinates
    y, x = np.ogrid[:size, :size]
    center = size / 2
    x = (x - center) / center
    y = (y - center) / center
    
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Apply mask
    if mask is not None:
        rho_masked = np.where(mask, rho, np.nan)
    else:
        rho_masked = rho
    
    # Generate basis functions
    basis = []
    indices = []
    
    j = 1  # Noll index
    for n in range(max_order + 1):
        for m in range(-n, n + 1, 2):
            if (n - abs(m)) % 2 == 0:
                Z = zernike_polynomial(n, m, rho_masked, theta)
                basis.append(Z)
                indices.append((j, n, m))
                j += 1
    
    return basis, indices

class ZernikeFitter:
    """Fit Zernike polynomials to wavefront data."""
    
    def __init__(self, max_order=6):
        self.max_order = max_order
        self.basis = None
        self.indices = None
        self.coefficients = None
        self.excluded_terms = []
        
    def generate_basis(self, wavefront, mask=None):
        """Generate Zernike basis for given wavefront size."""
        size = wavefront.shape[0]
        self.basis, self.indices = generate_zernike_basis(size, self.max_order, mask)
        
    def fit(self, wavefront, mask=None, exclude_terms=None):
        """
        Fit Zernike polynomials to wavefront.
        
        Args:
            wavefront: Wavefront data (2D array)
            mask: Binary mask (optional)
            exclude_terms: List of Noll indices to exclude (e.g., [1, 2, 3])
        
        Returns:
            coefficients: Zernike coefficients
        """
        if self.basis is None:
            self.generate_basis(wavefront, mask)
        
        if exclude_terms is None:
            exclude_terms = []
        self.excluded_terms = exclude_terms
        
        # Prepare data
        if mask is not None:
            valid = mask.astype(bool)
            w_flat = wavefront[valid]
        else:
            valid = np.ones_like(wavefront, dtype=bool)
            w_flat = wavefront.flatten()
        
        # Build design matrix (exclude specified terms)
        A_list = []
        for i, (j, n, m) in enumerate(self.indices):
            if j not in exclude_terms:
                z_flat = self.basis[i][valid] if mask is not None else self.basis[i].flatten()
                A_list.append(z_flat)
        
        A = np.column_stack(A_list)
        
        # Least squares fit
        coeffs_reduced, _, _, _ = np.linalg.lstsq(A, w_flat, rcond=None)
        
        # Insert zeros for excluded terms
        coeffs_full = []
        reduced_idx = 0
        for j, n, m in self.indices:
            if j in exclude_terms:
                coeffs_full.append(0.0)
            else:
                coeffs_full.append(coeffs_reduced[reduced_idx])
                reduced_idx += 1
        
        self.coefficients = np.array(coeffs_full)
        
        return self.coefficients
    
    def reconstruct(self, coefficients=None):
        """Reconstruct wavefront from Zernike coefficients."""
        if coefficients is None:
            coefficients = self.coefficients
        
        if self.basis is None or coefficients is None:
            return None
        
        reconstructed = np.zeros_like(self.basis[0])
        for i, coeff in enumerate(coefficients):
            reconstructed += coeff * self.basis[i]
        
        return reconstructed
    
    def get_term_names(self):
        """Get descriptive names for Zernike terms."""
        names = []
        for j, n, m in self.indices:
            name = zernike_name(n, m)
            names.append(f"Z{j}: {name}")
        return names

def zernike_name(n, m):
    """Get descriptive name for Zernike term."""
    names = {
        (0, 0): "Piston",
        (1, -1): "Tilt Y",
        (1, 1): "Tilt X",
        (2, -2): "Astigmatism 45°",
        (2, 0): "Defocus",
        (2, 2): "Astigmatism 0°",
        (3, -3): "Trefoil Y",
        (3, -1): "Coma Y",
        (3, 1): "Coma X",
        (3, 3): "Trefoil X",
        (4, -4): "Quadrafoil Y",
        (4, -2): "Secondary Astigmatism Y",
        (4, 0): "Spherical",
        (4, 2): "Secondary Astigmatism X",
        (4, 4): "Quadrafoil X",
    }
    return names.get((n, m), f"Z({n},{m})")
```

**Zernike Panel UI:**
```
┌──────────────────────────────────────────┐
│ Zernike Analysis                         │
│                                          │
│ Max Order: [6▼]  [Fit] [Refit]          │
│                                          │
│ Term Exclusion:                          │
│ □ Z1: Piston      □ Z2: Tilt X          │
│ □ Z3: Tilt Y      □ Z4: Defocus         │
│ □ Z5: Astig 0°    □ Z6: Astig 45°       │
│ ...                                      │
│                                          │
│ Preset: [Exclude Piston+Tilt ▼]         │
│                                          │
│ Coefficients (waves):                    │
│ [Bar Chart Display]                      │
│                                          │
│ Fit Quality:                             │
│ Residual RMS: 0.025 λ                    │
│                                          │
│ [View Reconstructed] [View Residual]     │
│ [Export Coefficients]                    │
└──────────────────────────────────────────┘
```

---

## Implementation Workflow

### Step 1: Project Setup
```bash
# Create project structure
mkdir -p interferogram_analyzer/{config,gui/{tabs,widgets,dialogs},core,utils,algorithms,resources/{icons,styles,examples}}
cd interferogram_analyzer

# Create __init__.py files
find . -type d -exec touch {}/__init__.py \;

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Create Basic Application Structure

**main.py:**
```python
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from gui.main_window import MainWindow

def main():
    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("Interferogram Analyzer")
    app.setOrganizationName("OpticsSoftware")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
```

**gui/main_window.py:**
```python
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget, QMenuBar, QToolBar
from PyQt6.QtCore import Qt
from gui.tabs.loading_tab import LoadingTab
from gui.tabs.editing_tab import EditingTab
from gui.tabs.analysis_tab import AnalysisTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interferogram Analyzer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Shared data between tabs
        self.current_image = None
        self.mask = None
        self.processed_image = None
        self.wavelength = 632.8e-9  # Default HeNe wavelength
        
        self.setup_ui()
        self.create_menu_bar()
        self.create_toolbar()
        
    def setup_ui(self):
        """Setup main UI with tabs."""
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
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
        
        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Dark Theme", lambda: self.set_theme('dark'))
        view_menu.addAction("Light Theme", lambda: self.set_theme('light'))
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("Documentation", self.show_help)
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
        
    def enable_tab(self, tab_index):
        """Enable specific tab."""
        self.tabs.setTabEnabled(tab_index, True)
        
    def switch_to_tab(self, tab_index):
        """Switch to specific tab."""
        self.tabs.setCurrentIndex(tab_index)
        
    def set_theme(self, theme):
        """Load and apply theme stylesheet."""
        # TODO: Implement theme loading
        pass
        
    def save_project(self):
        """Save current project state."""
        # TODO: Implement project saving
        pass
        
    def export_results(self):
        """Export analysis results."""
        # TODO: Implement results export
        pass
        
    def show_help(self):
        """Show help documentation."""
        # TODO: Implement help dialog
        pass
        
    def show_about(self):
        """Show about dialog."""
        # TODO: Implement about dialog
        pass
```

### Step 3: Implement Tab 1 (Loading & Masking)

**gui/tabs/loading_tab.py:**
```python
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSpinBox, QCheckBox, QGroupBox, QSplitter)
from PyQt6.QtCore import Qt
from gui.widgets.mask_editor import MaskEditor
from core.image_loader import load_image
import numpy as np

class LoadingTab(QWidget):
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
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_mask)
        
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
        
        # Outer radius
        outer_layout = QHBoxLayout()
        outer_layout.addWidget(QLabel("Outer Radius:"))
        self.outer_radius_spin = QSpinBox()
        self.outer_radius_spin.setRange(0, 2000)
        self.outer_radius_spin.setValue(200)
        self.outer_radius_spin.valueChanged.connect(self.update_mask_from_controls)
        outer_layout.addWidget(self.outer_radius_spin)
        mask_layout.addLayout(outer_layout)
        
        # Inner radius
        inner_layout = QHBoxLayout()
        inner_layout.addWidget(QLabel("Inner Radius:"))
        self.inner_radius_spin = QSpinBox()
        self.inner_radius_spin.setRange(0, 2000)
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
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Interferogram",
            "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp);;All Files (*)"
        )
        
        if filename:
            self.image = load_image(filename)
            if self.image is not None:
                self.mask_editor.set_image(self.image)
                
                # Update controls
                h, w = self.image.shape[:2]
                self.center_x_spin.setValue(w // 2)
                self.center_y_spin.setValue(h // 2)
                self.outer_radius_spin.setMaximum(min(h, w) // 2)
                self.outer_radius_spin.setValue(min(h, w) // 2 - 10)
                self.inner_radius_spin.setMaximum(min(h, w) // 2)
                self.inner_radius_spin.setValue(20)
                
                # Update info
                bit_depth = self.image.dtype
                self.info_label.setText(
                    f"Size: {w}x{h}\n"
                    f"Type: {bit_depth}\n"
                    f"Min: {self.image.min()}\n"
                    f"Max: {self.image.max()}"
                )
                
                self.apply_btn.setEnabled(True)
                
    def auto_detect_mask(self):
        """Auto-detect circular mask from image."""
        if self.image is None:
            return
        
        from core.masking import auto_detect_circle
        
        outer_params, inner_params = auto_detect_circle(self.image)
        
        if outer_params:
            cx, cy, r = outer_params
            self.center_x_spin.setValue(int(cx))
            self.center_y_spin.setValue(int(cy))
            self.outer_radius_spin.setValue(int(r))
            
        if inner_params:
            _, _, r_inner = inner_params
            self.inner_radius_spin.setValue(int(r_inner))
            
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
        
    def update_mask_from_controls(self):
        """Update mask editor from spinbox controls."""
        outer_r = self.outer_radius_spin.value()
        inner_r = self.inner_radius_spin.value()
        cx = self.center_x_spin.value()
        cy = self.center_y_spin.value()
        
        self.mask_editor.set_mask_parameters(outer_r, inner_r, cx, cy)
        
    def apply_mask(self):
        """Apply mask and proceed to editing tab."""
        if self.image is None:
            return
        
        # Generate mask
        self.mask = self.mask_editor.generate_mask()
        
        # Store in main window
        self.main_window.current_image = self.image
        self.main_window.mask = self.mask
        
        # Enable editing tab and switch to it
        self.main_window.enable_tab(1)
        self.main_window.switch_to_tab(1)
        
        # Initialize editing tab with image
        self.main_window.editing_tab.set_image(self.image, self.mask)
```

### Step 4: Continue Implementation...

The remaining tabs (Editing and Analysis) follow similar patterns:

1. **EditingTab** - implements all preprocessing tools with history management
2. **AnalysisTab** - contains nested tabs for different visualizations
3. Each widget is self-contained with clear interfaces

---

## Testing Strategy

### Unit Tests
- Test each core algorithm independently
- Test mask generation
- Test phase extraction with synthetic data
- Test Zernike fitting accuracy

### Integration Tests
- Test data flow between tabs
- Test undo/redo functionality
- Test export functionality

### Test Data
- Create synthetic interferograms with known wavefront errors
- Use real interferograms from various sources
- Test edge cases (large obscurations, high noise, etc.)

---

## Performance Considerations

1. **Large Images**
   - Implement image decimation for preview
   - Use efficient NumPy operations
   - Consider multiprocessing for heavy computations

2. **Real-time Preview**
   - Debounce parameter changes
   - Use lower resolution for preview
   - Implement cancel operation

3. **Memory Management**
   - Limit history size
   - Clear large arrays when not needed
   - Use memory-mapped arrays for very large data

---

## Future Enhancements

1. **Batch Processing**
   - Process multiple interferograms
   - Automated analysis pipeline

2. **Advanced Algorithms**
   - Multiple wavelength support
   - Temporal phase unwrapping
   - Machine learning denoising

3. **Reporting**
   - Automated report generation (PDF)
   - Custom templates
   - Batch export

4. **Calibration**
   - System calibration tools
   - Reference wavefront subtraction
   - Environmental correction

---

## Development Notes

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Document all functions with docstrings
- Keep functions focused and small

### Version Control
- Use Git for version control
- Commit frequently with clear messages
- Tag releases with semantic versioning

### Documentation
- Maintain README.md with usage instructions
- Create API documentation for core modules
- Provide example notebooks/scripts

---

## Appendix: Quick Reference

### Common Algorithms

**Circular Mask Generation:**
```python
y, x = np.ogrid[:h, :w]
distance = np.sqrt((x - cx)**2 + (y - cy)**2)
mask = (distance >= inner_r) & (distance <= outer_r)
```

**FFT Spectrum Display:**
```python
fft = np.fft.fft2(image)
fft_shifted = np.fft.fftshift(fft)
magnitude = np.log(1 + np.abs(fft_shifted))
```

**Phase Unwrapping:**
```python
from skimage.restoration import unwrap_phase
unwrapped = unwrap_phase(wrapped_phase)
```

**Zernike Fitting (least squares):**
```python
# A: design matrix (each column is a Zernike polynomial)
# b: wavefront data (flattened)
coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
```

---

## Summary

This comprehensive guide provides:
- Complete file structure
- Detailed feature specifications for all 3 tabs
- Core algorithm implementations
- UI design patterns
- Testing and deployment strategy

Start with basic structure, implement Tab 1 first, then progressively add features from Tab 2 and Tab 3. The modular design allows parallel development of different components.

**Recommended Development Order:**
1. Basic app structure + Tab 1 (1 week)
2. Tab 2 editing tools (1 week)
3. Tab 3 phase extraction & unwrapping (1 week)
4. Tab 3 Zernike & PSF (1 week)
5. Polish, testing, documentation (1 week)

Total estimated time: 5 weeks for full implementation.
