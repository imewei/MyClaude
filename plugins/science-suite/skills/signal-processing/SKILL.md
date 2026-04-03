---
name: signal-processing
description: "Implement signal processing with SciPy and NumPy including FFT analysis, digital filtering (FIR/IIR), wavelet transforms, spectral estimation, and signal conditioning. Use when processing sensor data, implementing frequency analysis, designing filters, or performing spectral analysis."
---

# Signal Processing

Analyze, filter, and transform signals in time and frequency domains.

## Expert Agent

For high-performance numerical computing with JAX backends, delegate to the expert agent:

- **`jax-pro`**: JAX computing specialist for JIT compilation, vectorization, and GPU-accelerated numerics.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`
  - *Capabilities*: Functional transforms (jit, vmap, pmap), Flax neural networks, Optax optimization.

## FFT Analysis

```python
import numpy as np
from scipy import fft

def compute_fft(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute single-sided FFT magnitude spectrum."""
    n = len(signal)
    yf = fft.rfft(signal)
    xf = fft.rfftfreq(n, d=1.0 / fs)
    magnitude = 2.0 / n * np.abs(yf)
    return xf, magnitude

def power_spectral_density(signal: np.ndarray, fs: float, nperseg: int = 256) -> tuple:
    """Welch's PSD estimate for robust spectral analysis."""
    from scipy.signal import welch
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, psd

# Example: extract dominant frequency
def dominant_frequency(signal: np.ndarray, fs: float) -> float:
    """Find the frequency with highest spectral power."""
    freqs, mag = compute_fft(signal, fs)
    return freqs[np.argmax(mag[1:]) + 1]  # Skip DC component
```

## Digital Filter Design

```python
from scipy.signal import butter, cheby1, firwin, sosfilt, sosfiltfilt

def design_butterworth(
    order: int, cutoff: float, fs: float, btype: str = "low"
) -> np.ndarray:
    """Design Butterworth filter, return second-order sections."""
    sos = butter(order, cutoff, btype=btype, fs=fs, output="sos")
    return sos

def design_fir(
    numtaps: int, cutoff: float, fs: float, window: str = "hamming"
) -> np.ndarray:
    """Design FIR lowpass filter."""
    return firwin(numtaps, cutoff, fs=fs, window=window)

def apply_filter(
    signal: np.ndarray, sos: np.ndarray, zero_phase: bool = True
) -> np.ndarray:
    """Apply SOS filter. Use zero-phase for offline analysis."""
    if zero_phase:
        return sosfiltfilt(sos, signal)
    return sosfilt(sos, signal)
```

## Filter Type Reference

| Filter | Order | Passband | Stopband | Phase | Use Case |
|--------|-------|----------|----------|-------|----------|
| Butterworth | N | Maximally flat | Gradual | Smooth | General purpose |
| Chebyshev I | N | Ripple | Steeper | Non-linear | Sharp cutoff needed |
| Chebyshev II | N | Flat | Ripple | Non-linear | Stopband rejection |
| Elliptic | N | Ripple | Ripple | Non-linear | Minimum order |
| FIR | M | Flexible | Flexible | Linear | Phase-sensitive |

## Windowing Functions

```python
from scipy.signal import get_window

def apply_window(signal: np.ndarray, window_type: str = "hann") -> np.ndarray:
    """Apply window function before FFT to reduce spectral leakage."""
    window = get_window(window_type, len(signal))
    return signal * window

# Common windows and their properties:
# hann      - Good frequency resolution, moderate leakage
# hamming   - Slightly better sidelobe suppression than Hann
# blackman  - Excellent sidelobe suppression, wider main lobe
# kaiser(beta) - Adjustable: beta=0 (rectangular) to beta=14 (narrow)
# flattop   - Best amplitude accuracy for calibration
```

## Wavelet Transform

```python
import pywt

def wavelet_decompose(
    signal: np.ndarray, wavelet: str = "db4", level: int = 5
) -> dict:
    """Multi-level wavelet decomposition."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return {
        "approximation": coeffs[0],
        "details": coeffs[1:],
        "levels": level,
        "wavelet": wavelet,
    }

def wavelet_denoise(
    signal: np.ndarray, wavelet: str = "db4", level: int = 4, threshold: str = "soft"
) -> np.ndarray:
    """Denoise signal using wavelet thresholding."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # MAD estimator
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, uthresh, mode=threshold))
    return pywt.waverec(denoised_coeffs, wavelet)
```

## Spectral Estimation Methods

```python
from scipy.signal import periodogram, welch, spectrogram

def compute_spectrogram(
    signal: np.ndarray, fs: float, nperseg: int = 256
) -> tuple:
    """Short-time Fourier transform for time-frequency analysis."""
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    return f, t, 10 * np.log10(Sxx + 1e-12)  # dB scale
```

## Resampling and Interpolation

```python
from scipy.signal import resample, decimate

def change_sample_rate(
    signal: np.ndarray, original_fs: float, target_fs: float
) -> np.ndarray:
    """Resample signal to new sample rate."""
    num_samples = int(len(signal) * target_fs / original_fs)
    return resample(signal, num_samples)

def downsample_safe(signal: np.ndarray, factor: int) -> np.ndarray:
    """Downsample with anti-aliasing filter."""
    return decimate(signal, factor, ftype="fir", zero_phase=True)
```

## Signal Processing Checklist

- [ ] Characterize signal: sample rate, duration, expected frequency content
- [ ] Check for DC offset and remove if necessary
- [ ] Apply appropriate window before spectral analysis
- [ ] Design filter with sufficient stopband attenuation (>40 dB)
- [ ] Use zero-phase filtering for offline analysis (sosfiltfilt)
- [ ] Verify filter stability (all poles inside unit circle for IIR)
- [ ] Check for aliasing: signal bandwidth < fs/2
- [ ] Validate results against known test signals
