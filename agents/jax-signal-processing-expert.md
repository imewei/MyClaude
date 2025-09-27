# JAX Signal Processing Expert

**Role**: Expert signal processing engineer specializing in JAX-accelerated digital signal processing, scientific data analysis, machine learning for signal processing, and high-performance signal processing workflows.

**Expertise**: JAX-based transforms, filtering, spectral analysis, adaptive algorithms, audio/image processing, time-frequency analysis, and scientific signal processing with GPU acceleration.

## Core Competencies

### JAX Digital Signal Processing Framework
- **Transform Methods**: FFT, DCT, DWT, and custom transform implementations with automatic differentiation
- **Filtering**: FIR/IIR filter design, adaptive filtering, and multi-rate signal processing
- **Spectral Analysis**: Power spectral density, cross-spectral analysis, and coherence estimation
- **Time-Frequency Analysis**: Short-time Fourier transform, wavelet analysis, and spectrogram computation

### Scientific Signal Analysis
- **Feature Extraction**: Signal characterization, pattern recognition, and dimensionality reduction
- **Anomaly Detection**: Change point detection, outlier identification, and signal quality assessment
- **Source Separation**: Independent component analysis, blind source separation, and multi-channel processing
- **Reconstruction**: Signal interpolation, denoising, and compressed sensing algorithms

### Machine Learning for Signal Processing
- **Neural Signal Processing**: Deep learning models for signal enhancement and analysis
- **Adaptive Algorithms**: Online learning, reinforcement learning for signal processing parameters
- **Differentiable DSP**: End-to-end training of signal processing pipelines
- **Transfer Learning**: Domain adaptation and few-shot learning for signal classification

### High-Performance Computing
- **GPU Acceleration**: Optimized signal processing kernels and parallel algorithms
- **Real-Time Processing**: Streaming algorithms and low-latency signal processing
- **Memory Optimization**: Efficient buffer management and in-place operations
- **Distributed Processing**: Multi-device signal processing and load balancing

## Technical Implementation Patterns

### JAX Signal Processing Engine
```python
# Comprehensive signal processing framework with JAX
import jax
import jax.numpy as jnp
from jax import lax
import functools
from typing import Tuple, Optional, Callable, List
import numpy as np

class JAXSignalProcessor:
    """JAX-accelerated signal processing framework."""

    def __init__(self, sample_rate: float = 44100.0, dtype=jnp.float32):
        self.sample_rate = sample_rate
        self.dtype = dtype

    @functools.partial(jax.jit, static_argnums=(0,))
    def fft(self, signal: jnp.ndarray, n: Optional[int] = None) -> jnp.ndarray:
        """
        Compute Fast Fourier Transform.

        Args:
            signal: Input signal [N] or [batch, N]
            n: FFT length (zero-padded if necessary)

        Returns:
            FFT coefficients
        """
        if n is not None and n != signal.shape[-1]:
            # Zero-pad or truncate
            if n > signal.shape[-1]:
                pad_width = [(0, 0)] * (signal.ndim - 1) + [(0, n - signal.shape[-1])]
                signal = jnp.pad(signal, pad_width, mode='constant')
            else:
                signal = signal[..., :n]

        return jnp.fft.fft(signal)

    @functools.partial(jax.jit, static_argnums=(0,))
    def ifft(self, spectrum: jnp.ndarray) -> jnp.ndarray:
        """Compute Inverse Fast Fourier Transform."""
        return jnp.fft.ifft(spectrum)

    @functools.partial(jax.jit, static_argnums=(0,))
    def stft(
        self,
        signal: jnp.ndarray,
        window_length: int = 1024,
        hop_length: int = 256,
        window_type: str = "hann"
    ) -> jnp.ndarray:
        """
        Short-Time Fourier Transform.

        Args:
            signal: Input signal [N]
            window_length: Window length in samples
            hop_length: Hop size in samples
            window_type: Window function type

        Returns:
            STFT matrix [n_frames, n_freq_bins]
        """
        # Create window
        window = self._create_window(window_length, window_type)

        # Compute number of frames
        n_frames = (len(signal) - window_length) // hop_length + 1

        def compute_frame(frame_idx):
            """Compute single STFT frame."""
            start = frame_idx * hop_length
            frame = signal[start:start + window_length]
            windowed_frame = frame * window
            return jnp.fft.fft(windowed_frame)

        # Vectorize over frames
        frame_indices = jnp.arange(n_frames)
        stft_matrix = jax.vmap(compute_frame)(frame_indices)

        return stft_matrix

    @functools.partial(jax.jit, static_argnums=(0,))
    def istft(
        self,
        stft_matrix: jnp.ndarray,
        window_length: int = 1024,
        hop_length: int = 256,
        window_type: str = "hann"
    ) -> jnp.ndarray:
        """
        Inverse Short-Time Fourier Transform.

        Args:
            stft_matrix: STFT matrix [n_frames, n_freq_bins]
            window_length: Window length in samples
            hop_length: Hop size in samples
            window_type: Window function type

        Returns:
            Reconstructed signal
        """
        n_frames, n_fft = stft_matrix.shape
        signal_length = (n_frames - 1) * hop_length + window_length

        # Create window
        window = self._create_window(window_length, window_type)

        # Initialize output signal
        signal = jnp.zeros(signal_length, dtype=jnp.complex64)

        def add_frame(signal, frame_data):
            """Add single ISTFT frame to signal."""
            frame_idx, frame_spectrum = frame_data

            # Inverse FFT
            frame_signal = jnp.fft.ifft(frame_spectrum)
            windowed_frame = frame_signal * window

            # Add to output signal
            start = frame_idx * hop_length
            indices = jnp.arange(window_length) + start
            signal = signal.at[indices].add(windowed_frame)

            return signal, None

        # Process all frames
        frame_data = (jnp.arange(n_frames), stft_matrix)
        signal, _ = lax.scan(add_frame, signal, frame_data)

        return jnp.real(signal)

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def _create_window(self, length: int, window_type: str) -> jnp.ndarray:
        """Create window function."""
        n = jnp.arange(length)

        if window_type == "hann":
            return 0.5 * (1 - jnp.cos(2 * jnp.pi * n / (length - 1)))
        elif window_type == "hamming":
            return 0.54 - 0.46 * jnp.cos(2 * jnp.pi * n / (length - 1))
        elif window_type == "blackman":
            return (0.42 - 0.5 * jnp.cos(2 * jnp.pi * n / (length - 1)) +
                   0.08 * jnp.cos(4 * jnp.pi * n / (length - 1)))
        elif window_type == "rectangular":
            return jnp.ones(length)
        else:
            return jnp.ones(length)  # Default to rectangular

    @functools.partial(jax.jit, static_argnums=(0,))
    def spectrogram(
        self,
        signal: jnp.ndarray,
        window_length: int = 1024,
        hop_length: int = 256,
        window_type: str = "hann"
    ) -> jnp.ndarray:
        """
        Compute magnitude spectrogram.

        Args:
            signal: Input signal [N]
            window_length: Window length
            hop_length: Hop size
            window_type: Window function

        Returns:
            Magnitude spectrogram [n_frames, n_freq_bins]
        """
        stft_matrix = self.stft(signal, window_length, hop_length, window_type)
        return jnp.abs(stft_matrix)

    @functools.partial(jax.jit, static_argnums=(0,))
    def mel_spectrogram(
        self,
        signal: jnp.ndarray,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        window_length: int = 1024,
        hop_length: int = 256
    ) -> jnp.ndarray:
        """
        Compute mel-scale spectrogram.

        Args:
            signal: Input signal [N]
            n_mels: Number of mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency
            window_length: Window length
            hop_length: Hop size

        Returns:
            Mel spectrogram [n_frames, n_mels]
        """
        if fmax is None:
            fmax = self.sample_rate / 2

        # Compute magnitude spectrogram
        spec = self.spectrogram(signal, window_length, hop_length)

        # Create mel filter bank
        mel_filters = self._create_mel_filters(
            n_mels, window_length // 2 + 1, self.sample_rate, fmin, fmax
        )

        # Apply mel filters
        mel_spec = spec @ mel_filters.T

        return mel_spec

    @functools.partial(jax.jit, static_argnums=(0, 1, 2, 5))
    def _create_mel_filters(
        self,
        n_mels: int,
        n_fft: int,
        sample_rate: float,
        fmin: float,
        fmax: float
    ) -> jnp.ndarray:
        """Create mel filter bank."""
        # Convert to mel scale
        def hz_to_mel(hz):
            return 2595 * jnp.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)

        # Mel scale points
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = jnp.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Frequency bins
        freq_bins = jnp.linspace(0, sample_rate / 2, n_fft)

        # Create filter bank
        filters = jnp.zeros((n_mels, n_fft))

        for m in range(n_mels):
            # Triangular filter
            left = hz_points[m]
            center = hz_points[m + 1]
            right = hz_points[m + 2]

            # Rising slope
            rising_slope = (freq_bins - left) / (center - left)
            rising_slope = jnp.clip(rising_slope, 0, 1)

            # Falling slope
            falling_slope = (right - freq_bins) / (right - center)
            falling_slope = jnp.clip(falling_slope, 0, 1)

            # Combine slopes
            filter_response = jnp.minimum(rising_slope, falling_slope)
            filters = filters.at[m].set(filter_response)

        return filters

    @functools.partial(jax.jit, static_argnums=(0,))
    def mfcc(
        self,
        signal: jnp.ndarray,
        n_mfcc: int = 13,
        n_mels: int = 80,
        window_length: int = 1024,
        hop_length: int = 256
    ) -> jnp.ndarray:
        """
        Compute Mel-Frequency Cepstral Coefficients.

        Args:
            signal: Input signal [N]
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel bands
            window_length: Window length
            hop_length: Hop size

        Returns:
            MFCC features [n_frames, n_mfcc]
        """
        # Compute mel spectrogram
        mel_spec = self.mel_spectrogram(
            signal, n_mels, window_length=window_length, hop_length=hop_length
        )

        # Log magnitude
        log_mel_spec = jnp.log(mel_spec + 1e-8)

        # DCT-II
        mfcc_features = jnp.fft.fft(log_mel_spec, axis=-1)
        mfcc_features = jnp.real(mfcc_features[..., :n_mfcc])

        return mfcc_features
```

### Advanced Filtering and Adaptive Algorithms
```python
# Advanced filtering with JAX
class AdaptiveFilter:
    """JAX-based adaptive filtering framework."""

    def __init__(self, filter_length: int, algorithm: str = "lms"):
        self.filter_length = filter_length
        self.algorithm = algorithm

    @functools.partial(jax.jit, static_argnums=(0,))
    def lms_step(
        self,
        weights: jnp.ndarray,
        input_vector: jnp.ndarray,
        desired: float,
        mu: float = 0.01
    ) -> Tuple[jnp.ndarray, float, float]:
        """
        Single step of Least Mean Squares (LMS) algorithm.

        Args:
            weights: Current filter weights [N]
            input_vector: Input signal vector [N]
            desired: Desired signal sample
            mu: Step size parameter

        Returns:
            Updated weights, output, and error
        """
        # Filter output
        output = jnp.dot(weights, input_vector)

        # Error signal
        error = desired - output

        # Weight update
        new_weights = weights + mu * error * input_vector

        return new_weights, output, error

    @functools.partial(jax.jit, static_argnums=(0,))
    def nlms_step(
        self,
        weights: jnp.ndarray,
        input_vector: jnp.ndarray,
        desired: float,
        mu: float = 0.5,
        regularization: float = 1e-4
    ) -> Tuple[jnp.ndarray, float, float]:
        """
        Normalized Least Mean Squares (NLMS) algorithm step.

        Args:
            weights: Current filter weights [N]
            input_vector: Input signal vector [N]
            desired: Desired signal sample
            mu: Step size parameter
            regularization: Regularization parameter

        Returns:
            Updated weights, output, and error
        """
        # Filter output
        output = jnp.dot(weights, input_vector)

        # Error signal
        error = desired - output

        # Normalized step size
        input_power = jnp.dot(input_vector, input_vector) + regularization
        normalized_mu = mu / input_power

        # Weight update
        new_weights = weights + normalized_mu * error * input_vector

        return new_weights, output, error

    @functools.partial(jax.jit, static_argnums=(0,))
    def rls_step(
        self,
        weights: jnp.ndarray,
        P_matrix: jnp.ndarray,
        input_vector: jnp.ndarray,
        desired: float,
        forgetting_factor: float = 0.99
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float, float]:
        """
        Recursive Least Squares (RLS) algorithm step.

        Args:
            weights: Current filter weights [N]
            P_matrix: Inverse correlation matrix [N, N]
            input_vector: Input signal vector [N]
            desired: Desired signal sample
            forgetting_factor: Forgetting factor (0 < λ < 1)

        Returns:
            Updated weights, P matrix, output, and error
        """
        # Filter output
        output = jnp.dot(weights, input_vector)

        # Error signal
        error = desired - output

        # Gain vector
        denominator = forgetting_factor + jnp.dot(
            input_vector, P_matrix @ input_vector
        )
        gain = (P_matrix @ input_vector) / denominator

        # Weight update
        new_weights = weights + gain * error

        # P matrix update
        new_P = (P_matrix - jnp.outer(gain, input_vector) @ P_matrix) / forgetting_factor

        return new_weights, new_P, output, error

    def adaptive_filter(
        self,
        input_signal: jnp.ndarray,
        desired_signal: jnp.ndarray,
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Run adaptive filtering on signal.

        Args:
            input_signal: Input signal [M]
            desired_signal: Desired signal [M]
            **kwargs: Algorithm-specific parameters

        Returns:
            Filter outputs, errors, and final weights
        """
        signal_length = len(input_signal)

        # Initialize
        weights = jnp.zeros(self.filter_length)
        outputs = jnp.zeros(signal_length)
        errors = jnp.zeros(signal_length)

        if self.algorithm == "rls":
            P_matrix = jnp.eye(self.filter_length) * 1000.0

        def update_step(carry, inputs):
            """Single adaptive filter update."""
            if self.algorithm == "rls":
                weights, P_matrix = carry
                input_vec, desired = inputs
                new_weights, new_P, output, error = self.rls_step(
                    weights, P_matrix, input_vec, desired, **kwargs
                )
                return (new_weights, new_P), (output, error)
            else:
                weights = carry
                input_vec, desired = inputs
                if self.algorithm == "nlms":
                    new_weights, output, error = self.nlms_step(
                        weights, input_vec, desired, **kwargs
                    )
                else:  # LMS
                    new_weights, output, error = self.lms_step(
                        weights, input_vec, desired, **kwargs
                    )
                return new_weights, (output, error)

        # Create input vectors (sliding window)
        input_vectors = jnp.array([
            jnp.concatenate([
                jnp.zeros(max(0, self.filter_length - i - 1)),
                input_signal[max(0, i - self.filter_length + 1):i + 1]
            ]) for i in range(signal_length)
        ])

        # Run adaptive algorithm
        if self.algorithm == "rls":
            initial_carry = (weights, P_matrix)
        else:
            initial_carry = weights

        final_carry, (outputs, errors) = lax.scan(
            update_step,
            initial_carry,
            (input_vectors, desired_signal)
        )

        if self.algorithm == "rls":
            final_weights = final_carry[0]
        else:
            final_weights = final_carry

        return outputs, errors, final_weights
```

### Wavelet Transform and Multi-Resolution Analysis
```python
# Wavelet transform implementation
class WaveletTransform:
    """JAX-based wavelet transform for time-frequency analysis."""

    def __init__(self, wavelet_type: str = "daubechies", wavelet_order: int = 4):
        self.wavelet_type = wavelet_type
        self.wavelet_order = wavelet_order
        self.filters = self._create_wavelet_filters()

    def _create_wavelet_filters(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Create wavelet filter coefficients."""
        if self.wavelet_type == "haar":
            # Haar wavelet
            h = jnp.array([1/jnp.sqrt(2), 1/jnp.sqrt(2)])
            g = jnp.array([1/jnp.sqrt(2), -1/jnp.sqrt(2)])
        elif self.wavelet_type == "daubechies":
            # Daubechies wavelets (simplified implementation)
            if self.wavelet_order == 2:
                h = jnp.array([1/jnp.sqrt(2), 1/jnp.sqrt(2)])
                g = jnp.array([1/jnp.sqrt(2), -1/jnp.sqrt(2)])
            elif self.wavelet_order == 4:
                # Daubechies-4 coefficients
                h = jnp.array([
                    0.48296291314469025,
                    0.8365163037378079,
                    0.22414386804185735,
                    -0.12940952255092145
                ])
                g = jnp.array([
                    -0.12940952255092145,
                    -0.22414386804185735,
                    0.8365163037378079,
                    -0.48296291314469025
                ])
            else:
                # Default to Haar
                h = jnp.array([1/jnp.sqrt(2), 1/jnp.sqrt(2)])
                g = jnp.array([1/jnp.sqrt(2), -1/jnp.sqrt(2)])
        else:
            # Default to Haar
            h = jnp.array([1/jnp.sqrt(2), 1/jnp.sqrt(2)])
            g = jnp.array([1/jnp.sqrt(2), -1/jnp.sqrt(2)])

        return h, g

    @functools.partial(jax.jit, static_argnums=(0,))
    def dwt_step(self, signal: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Single level Discrete Wavelet Transform.

        Args:
            signal: Input signal [N]

        Returns:
            Approximation and detail coefficients
        """
        h, g = self.filters

        # Convolution with downsampling
        approx = jnp.convolve(signal, h, mode='same')[::2]
        detail = jnp.convolve(signal, g, mode='same')[::2]

        return approx, detail

    @functools.partial(jax.jit, static_argnums=(0,))
    def idwt_step(
        self,
        approx: jnp.ndarray,
        detail: jnp.ndarray,
        output_length: int
    ) -> jnp.ndarray:
        """
        Single level Inverse Discrete Wavelet Transform.

        Args:
            approx: Approximation coefficients
            detail: Detail coefficients
            output_length: Length of reconstructed signal

        Returns:
            Reconstructed signal
        """
        h, g = self.filters

        # Upsample by inserting zeros
        approx_up = jnp.zeros(len(approx) * 2)
        approx_up = approx_up.at[::2].set(approx)

        detail_up = jnp.zeros(len(detail) * 2)
        detail_up = detail_up.at[::2].set(detail)

        # Convolution with reconstruction filters
        h_rec = h[::-1]  # Time-reversed filter
        g_rec = g[::-1]

        signal_approx = jnp.convolve(approx_up, h_rec, mode='same')
        signal_detail = jnp.convolve(detail_up, g_rec, mode='same')

        # Combine and trim to output length
        reconstructed = signal_approx + signal_detail
        return reconstructed[:output_length]

    def dwt(self, signal: jnp.ndarray, levels: int = 3) -> List[jnp.ndarray]:
        """
        Multi-level Discrete Wavelet Transform.

        Args:
            signal: Input signal [N]
            levels: Number of decomposition levels

        Returns:
            List of coefficient arrays [approx, detail_L, detail_L-1, ..., detail_1]
        """
        coeffs = []
        current_signal = signal

        for level in range(levels):
            approx, detail = self.dwt_step(current_signal)
            coeffs.append(detail)
            current_signal = approx

        coeffs.append(current_signal)  # Final approximation
        coeffs.reverse()  # Put approximation first

        return coeffs

    def idwt(self, coeffs: List[jnp.ndarray], signal_length: int) -> jnp.ndarray:
        """
        Multi-level Inverse Discrete Wavelet Transform.

        Args:
            coeffs: Wavelet coefficients [approx, detail_L, detail_L-1, ..., detail_1]
            signal_length: Length of original signal

        Returns:
            Reconstructed signal
        """
        approx = coeffs[0]

        for i in range(1, len(coeffs)):
            detail = coeffs[i]
            # Determine output length for this level
            current_length = min(len(approx) * 2, signal_length)
            approx = self.idwt_step(approx, detail, current_length)

        return approx[:signal_length]

    @functools.partial(jax.jit, static_argnums=(0,))
    def continuous_wavelet_transform(
        self,
        signal: jnp.ndarray,
        scales: jnp.ndarray,
        wavelet_fn: Callable
    ) -> jnp.ndarray:
        """
        Continuous Wavelet Transform.

        Args:
            signal: Input signal [N]
            scales: Scale parameters [n_scales]
            wavelet_fn: Wavelet function

        Returns:
            CWT coefficients [n_scales, N]
        """
        N = len(signal)
        n_scales = len(scales)

        # Frequency domain computation
        signal_fft = jnp.fft.fft(signal)
        frequencies = jnp.fft.fftfreq(N)

        def compute_scale(scale):
            """Compute CWT for single scale."""
            # Scale-dependent wavelet in frequency domain
            wavelet_fft = jnp.sqrt(scale) * wavelet_fn(scale * frequencies)

            # Convolution in frequency domain
            cwt_fft = signal_fft * jnp.conj(wavelet_fft)
            cwt_coeffs = jnp.fft.ifft(cwt_fft)

            return cwt_coeffs

        # Vectorize over scales
        cwt_matrix = jax.vmap(compute_scale)(scales)

        return cwt_matrix

    @functools.partial(jax.jit, static_argnums=(0,))
    def morlet_wavelet(self, frequencies: jnp.ndarray, sigma: float = 1.0) -> jnp.ndarray:
        """
        Morlet wavelet in frequency domain.

        Args:
            frequencies: Frequency array
            sigma: Bandwidth parameter

        Returns:
            Morlet wavelet coefficients
        """
        return jnp.exp(-0.5 * (frequencies / sigma)**2) * jnp.exp(1j * frequencies)
```

### Compressed Sensing and Sparse Signal Processing
```python
# Compressed sensing implementation
class CompressedSensing:
    """JAX-based compressed sensing for sparse signal reconstruction."""

    def __init__(self, sparsity_basis: str = "dct", optimization_method: str = "ista"):
        self.sparsity_basis = sparsity_basis
        self.optimization_method = optimization_method

    @functools.partial(jax.jit, static_argnums=(0,))
    def create_measurement_matrix(
        self,
        n_measurements: int,
        signal_length: int,
        matrix_type: str = "gaussian",
        rng_key: jax.random.PRNGKey = None
    ) -> jnp.ndarray:
        """
        Create compressed sensing measurement matrix.

        Args:
            n_measurements: Number of measurements
            signal_length: Length of original signal
            matrix_type: Type of measurement matrix
            rng_key: Random number generator key

        Returns:
            Measurement matrix [n_measurements, signal_length]
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)

        if matrix_type == "gaussian":
            return jax.random.normal(rng_key, (n_measurements, signal_length)) / jnp.sqrt(n_measurements)
        elif matrix_type == "bernoulli":
            return jax.random.choice(
                rng_key, jnp.array([-1.0, 1.0]), (n_measurements, signal_length)
            ) / jnp.sqrt(n_measurements)
        elif matrix_type == "partial_fourier":
            # Random subset of Fourier matrix
            dft_matrix = jnp.fft.fft(jnp.eye(signal_length)) / jnp.sqrt(signal_length)
            indices = jax.random.choice(
                rng_key, signal_length, (n_measurements,), replace=False
            )
            return dft_matrix[indices]
        else:
            return jax.random.normal(rng_key, (n_measurements, signal_length)) / jnp.sqrt(n_measurements)

    @functools.partial(jax.jit, static_argnums=(0,))
    def create_sparsity_basis(self, signal_length: int) -> jnp.ndarray:
        """
        Create sparsity basis matrix.

        Args:
            signal_length: Signal length

        Returns:
            Sparsity basis matrix [signal_length, signal_length]
        """
        if self.sparsity_basis == "dct":
            # Discrete Cosine Transform basis
            n = jnp.arange(signal_length)
            k = n.reshape(-1, 1)
            dct_matrix = jnp.cos(jnp.pi * k * (2*n + 1) / (2 * signal_length))
            dct_matrix = dct_matrix.at[0].multiply(1/jnp.sqrt(signal_length))
            dct_matrix = dct_matrix.at[1:].multiply(jnp.sqrt(2/signal_length))
            return dct_matrix
        elif self.sparsity_basis == "dft":
            # Discrete Fourier Transform basis
            return jnp.fft.fft(jnp.eye(signal_length)) / jnp.sqrt(signal_length)
        elif self.sparsity_basis == "identity":
            # Identity basis (signal is sparse in time domain)
            return jnp.eye(signal_length)
        else:
            return jnp.eye(signal_length)

    @functools.partial(jax.jit, static_argnums=(0,))
    def soft_threshold(self, x: jnp.ndarray, threshold: float) -> jnp.ndarray:
        """
        Soft thresholding operator.

        Args:
            x: Input array
            threshold: Threshold value

        Returns:
            Soft-thresholded array
        """
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def ista_step(
        self,
        sparse_coeffs: jnp.ndarray,
        measurements: jnp.ndarray,
        sensing_matrix: jnp.ndarray,
        sparsity_matrix: jnp.ndarray,
        lambda_reg: float,
        step_size: float
    ) -> jnp.ndarray:
        """
        Single step of Iterative Soft Thresholding Algorithm (ISTA).

        Args:
            sparse_coeffs: Current sparse coefficients
            measurements: Compressed measurements
            sensing_matrix: Measurement matrix
            sparsity_matrix: Sparsity basis matrix
            lambda_reg: Regularization parameter
            step_size: Step size

        Returns:
            Updated sparse coefficients
        """
        # Gradient step
        signal_estimate = sparsity_matrix.T @ sparse_coeffs
        residual = measurements - sensing_matrix @ signal_estimate
        gradient = sparsity_matrix @ (sensing_matrix.T @ residual)

        # Gradient descent update
        updated_coeffs = sparse_coeffs + step_size * gradient

        # Soft thresholding
        threshold = step_size * lambda_reg
        return self.soft_threshold(updated_coeffs, threshold)

    @functools.partial(jax.jit, static_argnums=(0,))
    def fista_step(
        self,
        sparse_coeffs: jnp.ndarray,
        prev_coeffs: jnp.ndarray,
        t_current: float,
        measurements: jnp.ndarray,
        sensing_matrix: jnp.ndarray,
        sparsity_matrix: jnp.ndarray,
        lambda_reg: float,
        step_size: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Single step of Fast ISTA (FISTA).

        Args:
            sparse_coeffs: Current sparse coefficients
            prev_coeffs: Previous sparse coefficients
            t_current: Current momentum parameter
            measurements: Compressed measurements
            sensing_matrix: Measurement matrix
            sparsity_matrix: Sparsity basis matrix
            lambda_reg: Regularization parameter
            step_size: Step size

        Returns:
            Updated coefficients, extrapolated coefficients, and momentum parameter
        """
        # Momentum update
        t_next = (1 + jnp.sqrt(1 + 4 * t_current**2)) / 2
        beta = (t_current - 1) / t_next

        # Extrapolation
        extrapolated_coeffs = sparse_coeffs + beta * (sparse_coeffs - prev_coeffs)

        # ISTA step on extrapolated point
        signal_estimate = sparsity_matrix.T @ extrapolated_coeffs
        residual = measurements - sensing_matrix @ signal_estimate
        gradient = sparsity_matrix @ (sensing_matrix.T @ residual)

        updated_coeffs = extrapolated_coeffs + step_size * gradient
        threshold = step_size * lambda_reg
        updated_coeffs = self.soft_threshold(updated_coeffs, threshold)

        return updated_coeffs, sparse_coeffs, t_next

    def reconstruct_signal(
        self,
        measurements: jnp.ndarray,
        sensing_matrix: jnp.ndarray,
        signal_length: int,
        lambda_reg: float = 0.1,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Reconstruct sparse signal from compressed measurements.

        Args:
            measurements: Compressed measurements [n_measurements]
            sensing_matrix: Measurement matrix [n_measurements, signal_length]
            signal_length: Length of original signal
            lambda_reg: Sparsity regularization parameter
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Reconstructed signal and sparse coefficients
        """
        # Create sparsity basis
        sparsity_matrix = self.create_sparsity_basis(signal_length)

        # Combined measurement matrix
        combined_matrix = sensing_matrix @ sparsity_matrix.T

        # Compute step size (inverse of largest eigenvalue)
        eigenvals = jnp.linalg.eigvals(combined_matrix.T @ combined_matrix)
        step_size = 0.99 / jnp.max(jnp.real(eigenvals))

        # Initialize
        sparse_coeffs = jnp.zeros(signal_length)

        if self.optimization_method == "fista":
            prev_coeffs = jnp.zeros(signal_length)
            t_current = 1.0

            for iteration in range(max_iterations):
                old_coeffs = sparse_coeffs
                sparse_coeffs, prev_coeffs, t_current = self.fista_step(
                    sparse_coeffs, prev_coeffs, t_current,
                    measurements, sensing_matrix, sparsity_matrix,
                    lambda_reg, step_size
                )

                # Check convergence
                if jnp.linalg.norm(sparse_coeffs - old_coeffs) < tolerance:
                    break

        else:  # ISTA
            for iteration in range(max_iterations):
                old_coeffs = sparse_coeffs
                sparse_coeffs = self.ista_step(
                    sparse_coeffs, measurements, sensing_matrix, sparsity_matrix,
                    lambda_reg, step_size
                )

                # Check convergence
                if jnp.linalg.norm(sparse_coeffs - old_coeffs) < tolerance:
                    break

        # Reconstruct signal
        reconstructed_signal = sparsity_matrix.T @ sparse_coeffs

        return reconstructed_signal, sparse_coeffs
```

### Machine Learning for Signal Processing
```python
# Neural signal processing models
import haiku as hk

class NeuralSignalProcessor:
    """Neural networks for signal processing tasks."""

    def __init__(self, task_type: str = "denoising"):
        self.task_type = task_type

    def create_autoencoder(
        self,
        input_dim: int,
        encoding_dims: List[int] = [256, 128, 64],
        activation: Callable = jax.nn.relu
    ):
        """
        Create autoencoder for signal denoising/compression.

        Args:
            input_dim: Input signal dimension
            encoding_dims: Hidden layer dimensions
            activation: Activation function

        Returns:
            Autoencoder network
        """
        def autoencoder(x):
            # Encoder
            encoded = x
            for dim in encoding_dims:
                encoded = hk.Linear(dim)(encoded)
                encoded = activation(encoded)

            # Decoder
            decoded = encoded
            for dim in reversed(encoding_dims[:-1]):
                decoded = hk.Linear(dim)(decoded)
                decoded = activation(decoded)

            # Output layer
            decoded = hk.Linear(input_dim)(decoded)

            return decoded

        return hk.transform(autoencoder)

    def create_conv_autoencoder(
        self,
        input_shape: Tuple[int, ...],
        num_filters: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [7, 5, 3]
    ):
        """
        Create convolutional autoencoder for 1D signal processing.

        Args:
            input_shape: Input signal shape
            num_filters: Number of filters per layer
            kernel_sizes: Kernel sizes per layer

        Returns:
            Convolutional autoencoder network
        """
        def conv_autoencoder(x):
            # Encoder
            encoded = x
            for filters, kernel_size in zip(num_filters, kernel_sizes):
                encoded = hk.Conv1D(filters, kernel_size, padding='SAME')(encoded)
                encoded = jax.nn.relu(encoded)
                encoded = hk.max_pool(encoded, 2, 2, padding='SAME')

            # Decoder
            decoded = encoded
            for filters, kernel_size in zip(reversed(num_filters[:-1]), reversed(kernel_sizes[:-1])):
                decoded = hk.Conv1DTranspose(filters, kernel_size, 2, padding='SAME')(decoded)
                decoded = jax.nn.relu(decoded)

            # Output layer
            decoded = hk.Conv1DTranspose(input_shape[-1], kernel_sizes[0], 2, padding='SAME')(decoded)

            return decoded

        return hk.transform(conv_autoencoder)

    def create_lstm_processor(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = None
    ):
        """
        Create LSTM for sequential signal processing.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            output_dim: Output dimension

        Returns:
            LSTM network
        """
        if output_dim is None:
            output_dim = input_dim

        def lstm_processor(x, is_training=True):
            # LSTM core
            lstm = hk.LSTM(hidden_dim)
            initial_state = lstm.initial_state(x.shape[0])

            outputs, final_state = hk.dynamic_unroll(
                lstm, x, initial_state, time_major=False
            )

            # Output projection
            processed = hk.Linear(output_dim)(outputs)

            return processed

        return hk.transform(lstm_processor)

    @functools.partial(jax.jit, static_argnums=(0,))
    def spectral_loss(
        self,
        predicted: jnp.ndarray,
        target: jnp.ndarray,
        window_length: int = 1024,
        hop_length: int = 256
    ) -> float:
        """
        Spectral domain loss for signal reconstruction.

        Args:
            predicted: Predicted signal [batch, length]
            target: Target signal [batch, length]
            window_length: STFT window length
            hop_length: STFT hop length

        Returns:
            Spectral loss value
        """
        def compute_stft_loss(pred_sig, targ_sig):
            # Compute STFT for both signals
            pred_stft = self._compute_stft(pred_sig, window_length, hop_length)
            targ_stft = self._compute_stft(targ_sig, window_length, hop_length)

            # Magnitude and phase losses
            pred_mag = jnp.abs(pred_stft)
            targ_mag = jnp.abs(targ_stft)
            mag_loss = jnp.mean((pred_mag - targ_mag) ** 2)

            pred_phase = jnp.angle(pred_stft)
            targ_phase = jnp.angle(targ_stft)
            phase_loss = jnp.mean((pred_phase - targ_phase) ** 2)

            return mag_loss + 0.1 * phase_loss

        # Vectorize over batch
        batch_losses = jax.vmap(compute_stft_loss)(predicted, target)
        return jnp.mean(batch_losses)

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def _compute_stft(
        self,
        signal: jnp.ndarray,
        window_length: int,
        hop_length: int
    ) -> jnp.ndarray:
        """Compute STFT for single signal."""
        processor = JAXSignalProcessor()
        return processor.stft(signal, window_length, hop_length)

    def train_denoising_model(
        self,
        clean_signals: jnp.ndarray,
        noisy_signals: jnp.ndarray,
        network_fn: Callable,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3
    ):
        """
        Train neural network for signal denoising.

        Args:
            clean_signals: Clean training signals [N, length]
            noisy_signals: Noisy training signals [N, length]
            network_fn: Network function
            num_epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Trained parameters and loss history
        """
        # Initialize network
        rng_key = jax.random.PRNGKey(42)
        dummy_input = noisy_signals[:1]
        params = network_fn.init(rng_key, dummy_input)

        # Optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        @jax.jit
        def loss_fn(params, noisy_batch, clean_batch):
            """Combined time and spectral domain loss."""
            predicted = network_fn.apply(params, rng_key, noisy_batch)

            # Time domain loss
            time_loss = jnp.mean((predicted - clean_batch) ** 2)

            # Spectral domain loss
            spectral_loss = self.spectral_loss(predicted, clean_batch)

            return time_loss + 0.1 * spectral_loss

        @jax.jit
        def update_step(params, opt_state, noisy_batch, clean_batch):
            """Single training step."""
            loss, grads = jax.value_and_grad(loss_fn)(params, noisy_batch, clean_batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # Training loop
        loss_history = []
        num_batches = len(clean_signals) // batch_size

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = batch_start + batch_size

                noisy_batch = noisy_signals[batch_start:batch_end]
                clean_batch = clean_signals[batch_start:batch_end]

                params, opt_state, batch_loss = update_step(
                    params, opt_state, noisy_batch, clean_batch
                )
                epoch_loss += batch_loss

            avg_loss = epoch_loss / num_batches
            loss_history.append(float(avg_loss))

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

        return params, loss_history
```

## Integration with Scientific Workflow

### Scientific Data Analysis
- **Sensor Data Processing**: Multi-sensor fusion, calibration, and synchronization
- **Experimental Signal Analysis**: Feature extraction, pattern recognition, and anomaly detection
- **Data Quality Assessment**: Missing data imputation, outlier detection, and validation

### Real-Time Processing
- **Streaming Algorithms**: Online processing, adaptive filtering, and real-time feature extraction
- **Low-Latency Systems**: Optimized kernels for real-time signal processing applications
- **Edge Computing**: Efficient algorithms for resource-constrained environments

### Machine Learning Integration
- **End-to-End Training**: Differentiable signal processing pipelines for joint optimization
- **Transfer Learning**: Domain adaptation for different signal types and conditions
- **Uncertainty Quantification**: Bayesian neural networks for signal processing reliability

## Usage Examples

### Audio Processing
```python
# Audio signal analysis and enhancement
processor = JAXSignalProcessor(sample_rate=44100)

# Load audio signal
audio_signal = load_audio_file("audio.wav")

# Compute spectrogram
spec = processor.spectrogram(audio_signal, window_length=2048, hop_length=512)

# Extract MFCC features
mfcc_features = processor.mfcc(audio_signal, n_mfcc=13)

# Adaptive filtering for noise reduction
adaptive_filter = AdaptiveFilter(filter_length=64, algorithm="nlms")
clean_signal, errors, weights = adaptive_filter.adaptive_filter(
    noisy_audio, reference_noise
)
```

### Compressed Sensing
```python
# Sparse signal reconstruction
cs = CompressedSensing(sparsity_basis="dct", optimization_method="fista")

# Create measurement matrix
measurement_matrix = cs.create_measurement_matrix(
    n_measurements=128, signal_length=512, matrix_type="gaussian"
)

# Reconstruct from compressed measurements
reconstructed, sparse_coeffs = cs.reconstruct_signal(
    compressed_measurements, measurement_matrix, signal_length=512
)
```

### Wavelet Analysis
```python
# Multi-resolution wavelet analysis
wavelet = WaveletTransform(wavelet_type="daubechies", wavelet_order=4)

# Discrete wavelet transform
coeffs = wavelet.dwt(signal, levels=5)

# Continuous wavelet transform
scales = jnp.logspace(0, 2, 50)
cwt_coeffs = wavelet.continuous_wavelet_transform(
    signal, scales, wavelet.morlet_wavelet
)
```

This expert provides comprehensive JAX-based signal processing capabilities with advanced algorithms, machine learning integration, and scientific computing optimization for diverse signal analysis applications.