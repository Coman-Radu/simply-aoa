"""
Doppler Detection and Measurement Methods

This module provides various methods to detect and measure Doppler shifts in acoustic signals,
which can be used to improve DOA estimation for moving sources.

Methods included:
1. Spectral centroid tracking (FFT-based)
2. Instantaneous frequency via Hilbert transform
3. Time-frequency analysis (STFT)
4. Cross-channel Doppler differential
5. Autocorrelation-based frequency tracking
6. Multi-hypothesis tracking filter
"""

import numpy as np
from scipy.signal import hilbert, stft, spectrogram
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


class DopplerDetector:
    """
    Multi-method Doppler shift detector for acoustic signals.
    """

    def __init__(self, sample_rate=48000, expected_freq=None, freq_range=None):
        """
        Initialize Doppler detector.

        Args:
            sample_rate: Sample rate in Hz
            expected_freq: Expected carrier frequency (if known), Hz
            freq_range: Tuple (f_min, f_max) for frequency analysis
        """
        self.sample_rate = sample_rate
        self.expected_freq = expected_freq

        if freq_range is None:
            if expected_freq is not None:
                # ±10% around expected frequency
                self.freq_range = (expected_freq * 0.9, expected_freq * 1.1)
            else:
                # Default to drone range
                self.freq_range = (500, 8000)
        else:
            self.freq_range = freq_range

    def method1_spectral_centroid(self, signal, window_duration=0.05):
        """
        Method 1: Track frequency via spectral centroid.

        Simple and fast, good for narrowband signals with clear peaks.

        Args:
            signal: Audio signal
            window_duration: Analysis window duration in seconds

        Returns:
            times: Time points for each measurement
            frequencies: Estimated frequency at each time
            doppler_shifts: Doppler shift relative to expected frequency
        """
        window_samples = int(window_duration * self.sample_rate)
        hop_samples = window_samples // 2

        num_windows = (len(signal) - window_samples) // hop_samples + 1

        times = []
        frequencies = []

        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples

            if end > len(signal):
                break

            window = signal[start:end]

            # Apply Hann window
            window = window * np.hanning(len(window))

            # FFT
            spectrum = np.abs(fft(window))
            freqs = fftfreq(len(window), 1/self.sample_rate)

            # Only positive frequencies in range
            mask = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
            spectrum = spectrum[mask]
            freqs = freqs[mask]

            if len(spectrum) == 0:
                continue

            # Spectral centroid (weighted mean frequency)
            if np.sum(spectrum) > 0:
                centroid = np.sum(freqs * spectrum) / np.sum(spectrum)

                times.append(start / self.sample_rate)
                frequencies.append(centroid)

        times = np.array(times)
        frequencies = np.array(frequencies)

        # Calculate Doppler shift
        if self.expected_freq is not None:
            doppler_shifts = frequencies - self.expected_freq
        else:
            # Use mean as reference
            doppler_shifts = frequencies - np.mean(frequencies)

        return times, frequencies, doppler_shifts

    def method2_peak_tracking(self, signal, window_duration=0.05):
        """
        Method 2: Track dominant frequency peak in spectrum.

        More robust to noise than centroid, good for tonal signals.

        Args:
            signal: Audio signal
            window_duration: Analysis window duration in seconds

        Returns:
            times: Time points
            frequencies: Peak frequencies
            doppler_shifts: Doppler shifts
        """
        window_samples = int(window_duration * self.sample_rate)
        hop_samples = window_samples // 2

        num_windows = (len(signal) - window_samples) // hop_samples + 1

        times = []
        frequencies = []

        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples

            if end > len(signal):
                break

            window = signal[start:end]
            window = window * np.hanning(len(window))

            # FFT with zero-padding for better frequency resolution
            nfft = 4 * len(window)
            spectrum = np.abs(fft(window, n=nfft))
            freqs = fftfreq(nfft, 1/self.sample_rate)

            # Only positive frequencies in range
            mask = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
            spectrum = spectrum[mask]
            freqs = freqs[mask]

            if len(spectrum) == 0:
                continue

            # Find peak
            peak_idx = np.argmax(spectrum)
            peak_freq = freqs[peak_idx]

            times.append(start / self.sample_rate)
            frequencies.append(peak_freq)

        times = np.array(times)
        frequencies = np.array(frequencies)

        if self.expected_freq is not None:
            doppler_shifts = frequencies - self.expected_freq
        else:
            doppler_shifts = frequencies - np.median(frequencies)

        return times, frequencies, doppler_shifts

    def method3_hilbert_instantaneous_freq(self, signal, window_duration=0.05):
        """
        Method 3: Instantaneous frequency via Hilbert transform.

        Good for frequency-modulated signals, works well with Doppler.

        Args:
            signal: Audio signal
            window_duration: Smoothing window duration

        Returns:
            times: Time points
            frequencies: Instantaneous frequencies
            doppler_shifts: Doppler shifts
        """
        # Bandpass filter to frequency range (simple approach)
        from scipy.signal import butter, filtfilt

        nyquist = self.sample_rate / 2
        low = self.freq_range[0] / nyquist
        high = self.freq_range[1] / nyquist

        # Ensure valid range
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))

        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)

        # Hilbert transform to get analytic signal
        analytic_signal = hilbert(filtered)

        # Instantaneous phase
        phase = np.unwrap(np.angle(analytic_signal))

        # Instantaneous frequency = derivative of phase / 2π
        inst_freq = np.diff(phase) * self.sample_rate / (2 * np.pi)

        # Smooth with moving average
        window_samples = int(window_duration * self.sample_rate)
        if window_samples > 0:
            inst_freq = np.convolve(inst_freq, np.ones(window_samples)/window_samples, mode='valid')
            times = np.arange(len(inst_freq)) / self.sample_rate + window_duration/2
        else:
            times = np.arange(len(inst_freq)) / self.sample_rate

        if self.expected_freq is not None:
            doppler_shifts = inst_freq - self.expected_freq
        else:
            doppler_shifts = inst_freq - np.median(inst_freq)

        return times, inst_freq, doppler_shifts

    def method4_cross_channel_differential(self, signals, mic_positions):
        """
        Method 4: Differential Doppler across microphones.

        Measures Doppler difference between microphone pairs to infer velocity.
        This is unique because it directly relates to source motion direction.

        Args:
            signals: Array of signals [num_mics, num_samples]
            mic_positions: Array of mic positions [num_mics, 2] in meters

        Returns:
            doppler_gradients: Doppler gradient for each mic pair
            velocity_estimate: Estimated velocity vector [vx, vy]
        """
        num_mics = signals.shape[0]

        # Estimate frequency for each microphone using peak tracking
        mic_frequencies = []
        for mic_idx in range(num_mics):
            _, freqs, _ = self.method2_peak_tracking(signals[mic_idx], window_duration=0.1)
            mic_frequencies.append(np.mean(freqs))  # Average over time

        mic_frequencies = np.array(mic_frequencies)

        # Calculate Doppler differences between mic pairs
        # For each pair (i,j), the Doppler difference is related to:
        # Δf_ij = f_source * (v · r_ij) / (c * r_i * r_j)
        # where r_ij = r_i - r_j is the mic separation vector

        speed_of_sound = 343.0

        # Use reference frequency (mean across all mics)
        f_ref = np.mean(mic_frequencies)

        # Calculate Doppler shifts relative to reference
        doppler_shifts = mic_frequencies - f_ref

        # Estimate velocity using least-squares
        # doppler_shift[i] ≈ f_ref * (v · p_i) / (c * ||p_source||)
        # For far-field: doppler_shift[i] ≈ f_ref * (v · p_i) / (c * d)

        # Simplified: assume source is far, use mic position gradients
        if num_mics >= 3:
            # Fit plane to Doppler shifts vs position
            A = np.column_stack([mic_positions, np.ones(num_mics)])
            coeffs, residuals, rank, s = np.linalg.lstsq(A, doppler_shifts, rcond=None)

            # Gradient gives velocity direction (approximate)
            velocity_direction = coeffs[:2]

            # Scale to actual velocity (rough approximation)
            # |Δf| = f_ref * |v| / c
            max_doppler = np.max(np.abs(doppler_shifts))
            velocity_magnitude = (max_doppler * speed_of_sound) / f_ref

            if np.linalg.norm(velocity_direction) > 0:
                velocity_estimate = velocity_direction / np.linalg.norm(velocity_direction) * velocity_magnitude
            else:
                velocity_estimate = np.array([0.0, 0.0])
        else:
            velocity_estimate = np.array([0.0, 0.0])

        return doppler_shifts, velocity_estimate

    def method5_stft_ridge_tracking(self, signal, nperseg=512):
        """
        Method 5: Track frequency ridge in STFT spectrogram.

        Best for visualizing and tracking Doppler over time.

        Args:
            signal: Audio signal
            nperseg: STFT window length

        Returns:
            times: Time points
            frequencies: Tracked frequency ridge
            doppler_shifts: Doppler shifts
            spectrogram_data: (freqs, times, Sxx) for plotting
        """
        # Compute STFT
        freqs, times, Sxx = spectrogram(signal, self.sample_rate, nperseg=nperseg)

        # Only keep frequencies in range
        mask = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
        freqs = freqs[mask]
        Sxx = Sxx[mask, :]

        # Track ridge (maximum energy at each time)
        ridge_indices = np.argmax(Sxx, axis=0)
        ridge_freqs = freqs[ridge_indices]

        if self.expected_freq is not None:
            doppler_shifts = ridge_freqs - self.expected_freq
        else:
            doppler_shifts = ridge_freqs - np.median(ridge_freqs)

        return times, ridge_freqs, doppler_shifts, (freqs, times, Sxx)

    def method6_autocorrelation_tracking(self, signal, window_duration=0.05):
        """
        Method 6: Frequency estimation via autocorrelation.

        Robust to noise, good for periodic signals.

        Args:
            signal: Audio signal
            window_duration: Analysis window duration

        Returns:
            times: Time points
            frequencies: Estimated frequencies
            doppler_shifts: Doppler shifts
        """
        window_samples = int(window_duration * self.sample_rate)
        hop_samples = window_samples // 2

        num_windows = (len(signal) - window_samples) // hop_samples + 1

        times = []
        frequencies = []

        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples

            if end > len(signal):
                break

            window = signal[start:end]

            # Autocorrelation
            autocorr = np.correlate(window, window, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags

            # Find first peak after zero lag (fundamental period)
            # Search in expected frequency range
            min_lag = int(self.sample_rate / self.freq_range[1])
            max_lag = int(self.sample_rate / self.freq_range[0])

            if max_lag > len(autocorr):
                max_lag = len(autocorr) - 1

            search_region = autocorr[min_lag:max_lag]

            if len(search_region) > 0:
                peak_idx = np.argmax(search_region) + min_lag

                # Frequency = sample_rate / period
                frequency = self.sample_rate / peak_idx

                times.append(start / self.sample_rate)
                frequencies.append(frequency)

        times = np.array(times)
        frequencies = np.array(frequencies)

        if self.expected_freq is not None:
            doppler_shifts = frequencies - self.expected_freq
        else:
            doppler_shifts = frequencies - np.median(frequencies)

        return times, frequencies, doppler_shifts


class DopplerCompensator:
    """
    Methods to compensate for Doppler effects in TDOA/DOA estimation.
    """

    def __init__(self, sample_rate=48000, speed_of_sound=343.0):
        self.sample_rate = sample_rate
        self.speed_of_sound = speed_of_sound

    def resample_for_doppler(self, signal, doppler_shift, carrier_freq):
        """
        Resample signal to compensate for Doppler shift.

        Args:
            signal: Audio signal
            doppler_shift: Doppler shift in Hz
            carrier_freq: Original carrier frequency

        Returns:
            resampled: Doppler-compensated signal
        """
        from scipy.signal import resample_poly

        # Doppler ratio: f_observed / f_source
        doppler_ratio = (carrier_freq + doppler_shift) / carrier_freq

        # Resample to compensate
        # If ratio > 1 (compressed), stretch by resampling down
        # If ratio < 1 (stretched), compress by resampling up

        if abs(doppler_ratio - 1.0) < 0.001:
            return signal

        # Calculate resampling factors
        up = 1000
        down = int(1000 * doppler_ratio)

        resampled = resample_poly(signal, up, down)

        # Truncate or pad to original length
        if len(resampled) > len(signal):
            resampled = resampled[:len(signal)]
        else:
            resampled = np.pad(resampled, (0, len(signal) - len(resampled)))

        return resampled

    def adaptive_buffer_segmentation(self, signal, estimated_angular_velocity,
                                    azimuth_current, buffer_duration=0.5):
        """
        Segment buffer into sub-buffers based on estimated motion.

        Args:
            signal: Audio signal
            estimated_angular_velocity: Angular velocity in deg/s
            azimuth_current: Current azimuth estimate in degrees
            buffer_duration: Total buffer duration

        Returns:
            segments: List of (signal_segment, azimuth_estimate) tuples
        """
        # Calculate how much the source moves during buffer
        angular_displacement = abs(estimated_angular_velocity) * buffer_duration

        # Decide number of segments
        if angular_displacement < 10:
            num_segments = 1  # Stationary enough
        elif angular_displacement < 30:
            num_segments = 2
        elif angular_displacement < 60:
            num_segments = 4
        else:
            num_segments = 8

        segment_length = len(signal) // num_segments
        segments = []

        for i in range(num_segments):
            start = i * segment_length
            end = (i + 1) * segment_length if i < num_segments - 1 else len(signal)

            # Time at center of segment
            t_center = (start + end) / 2 / self.sample_rate

            # Estimated azimuth at this time
            azimuth_seg = azimuth_current + estimated_angular_velocity * t_center

            segments.append((signal[start:end], azimuth_seg))

        return segments

    def velocity_from_doppler_gradient(self, doppler_shifts, mic_positions, source_position_guess=None):
        """
        Estimate source velocity from Doppler measurements at multiple microphones.

        Args:
            doppler_shifts: Array of Doppler shifts at each mic [num_mics]
            mic_positions: Array of mic positions [num_mics, 2]
            source_position_guess: Estimated source position [x, y] (optional)

        Returns:
            velocity: Estimated velocity vector [vx, vy] in m/s
        """
        num_mics = len(doppler_shifts)

        if source_position_guess is None:
            # Assume far-field (source very distant)
            # In this case, Doppler shift is proportional to mic position

            # Linear regression: doppler = k * position + offset
            A = np.column_stack([mic_positions, np.ones(num_mics)])
            coeffs, _, _, _ = np.linalg.lstsq(A, doppler_shifts, rcond=None)

            # Gradient gives velocity direction
            velocity = coeffs[:2] * self.speed_of_sound
        else:
            # Near-field: account for actual geometry
            vectors_to_source = source_position_guess - mic_positions
            distances = np.linalg.norm(vectors_to_source, axis=1)
            unit_vectors = vectors_to_source / distances[:, np.newaxis]

            # doppler_shift[i] = -f_carrier * (v · r_hat_i) / c
            # Solve for v using least squares
            # This is underdetermined for 2D velocity with 1D measurements
            # Use pseudo-inverse

            # A * v = doppler_shifts * (-c / f_carrier)
            # Assuming f_carrier ≈ 3000 Hz
            f_carrier = 3000
            b = doppler_shifts * (-self.speed_of_sound / f_carrier)

            velocity, _, _, _ = np.linalg.lstsq(unit_vectors, b, rcond=None)

        return velocity


def demonstrate_doppler_detection():
    """
    Demonstrate all Doppler detection methods on simulated data.
    """
    print("=" * 80)
    print("DOPPLER DETECTION METHODS DEMONSTRATION")
    print("=" * 80)

    # Generate test signal with Doppler
    sample_rate = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Carrier frequency varies linearly (simulating constant acceleration)
    f_start = 2900  # Hz
    f_end = 3100    # Hz
    freq_trajectory = f_start + (f_end - f_start) * (t / duration)

    # Generate signal with time-varying frequency
    phase = 2 * np.pi * np.cumsum(freq_trajectory) / sample_rate
    signal = np.sin(phase)

    # Add noise
    signal += 0.1 * np.random.randn(len(signal))

    print(f"Test signal: {f_start} Hz → {f_end} Hz over {duration} s")
    print(f"Doppler shift: {f_end - f_start} Hz")
    print(f"Sample rate: {sample_rate} Hz")
    print()

    # Initialize detector
    detector = DopplerDetector(sample_rate=sample_rate, expected_freq=3000)

    # Method 1: Spectral centroid
    print("Method 1: Spectral Centroid Tracking...")
    t1, f1, d1 = detector.method1_spectral_centroid(signal)
    print(f"  Mean frequency: {np.mean(f1):.1f} Hz")
    print(f"  Doppler range: {np.min(d1):.1f} to {np.max(d1):.1f} Hz")

    # Method 2: Peak tracking
    print("\nMethod 2: Peak Tracking...")
    t2, f2, d2 = detector.method2_peak_tracking(signal)
    print(f"  Mean frequency: {np.mean(f2):.1f} Hz")
    print(f"  Doppler range: {np.min(d2):.1f} to {np.max(d2):.1f} Hz")

    # Method 3: Hilbert instantaneous frequency
    print("\nMethod 3: Hilbert Transform Instantaneous Frequency...")
    t3, f3, d3 = detector.method3_hilbert_instantaneous_freq(signal)
    print(f"  Mean frequency: {np.mean(f3):.1f} Hz")
    print(f"  Doppler range: {np.min(d3):.1f} to {np.max(d3):.1f} Hz")

    # Method 5: STFT ridge tracking
    print("\nMethod 5: STFT Ridge Tracking...")
    t5, f5, d5, spec_data = detector.method5_stft_ridge_tracking(signal)
    print(f"  Mean frequency: {np.mean(f5):.1f} Hz")
    print(f"  Doppler range: {np.min(d5):.1f} to {np.max(d5):.1f} Hz")

    # Method 6: Autocorrelation
    print("\nMethod 6: Autocorrelation Tracking...")
    t6, f6, d6 = detector.method6_autocorrelation_tracking(signal)
    print(f"  Mean frequency: {np.mean(f6):.1f} Hz")
    print(f"  Doppler range: {np.min(d6):.1f} to {np.max(d6):.1f} Hz")

    print()
    print("=" * 80)

    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Plot 1: Spectrogram with ridge
    ax = axes[0, 0]
    freqs_spec, times_spec, Sxx = spec_data
    ax.pcolormesh(times_spec, freqs_spec, 10 * np.log10(Sxx + 1e-10),
                  shading='gouraud', cmap='viridis')
    ax.plot(t5, f5, 'r-', linewidth=2, label='Tracked ridge')
    ax.plot(t, freq_trajectory, 'w--', linewidth=2, label='True frequency', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Method 5: STFT Ridge Tracking')
    ax.legend()
    ax.set_ylim(detector.freq_range)

    # Plot 2: Spectral centroid
    ax = axes[0, 1]
    ax.plot(t1, f1, 'b-', linewidth=2, label='Centroid')
    ax.plot(t, freq_trajectory, 'k--', linewidth=1, label='True', alpha=0.7)
    ax.axhline(3000, color='gray', linestyle=':', label='Expected')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Method 1: Spectral Centroid')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Peak tracking
    ax = axes[1, 0]
    ax.plot(t2, f2, 'g-', linewidth=2, label='Peak freq')
    ax.plot(t, freq_trajectory, 'k--', linewidth=1, label='True', alpha=0.7)
    ax.axhline(3000, color='gray', linestyle=':', label='Expected')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Method 2: Peak Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Hilbert instantaneous frequency
    ax = axes[1, 1]
    ax.plot(t3, f3, 'm-', linewidth=2, label='Inst. freq')
    ax.plot(t, freq_trajectory, 'k--', linewidth=1, label='True', alpha=0.7)
    ax.axhline(3000, color='gray', linestyle=':', label='Expected')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Method 3: Hilbert Instantaneous Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Autocorrelation
    ax = axes[2, 0]
    ax.plot(t6, f6, 'c-', linewidth=2, label='Autocorr freq')
    ax.plot(t, freq_trajectory, 'k--', linewidth=1, label='True', alpha=0.7)
    ax.axhline(3000, color='gray', linestyle=':', label='Expected')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Method 6: Autocorrelation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Comparison of Doppler shifts
    ax = axes[2, 1]
    ax.plot(t1, d1, 'b-', alpha=0.7, label='Centroid')
    ax.plot(t2, d2, 'g-', alpha=0.7, label='Peak')
    ax.plot(t3, d3, 'm-', alpha=0.7, label='Hilbert')
    ax.plot(t5, d5, 'r-', alpha=0.7, label='STFT ridge')
    ax.plot(t6, d6, 'c-', alpha=0.7, label='Autocorr')
    true_doppler = freq_trajectory - 3000
    ax.plot(t, true_doppler, 'k--', linewidth=2, label='True')
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Doppler Shift (Hz)')
    ax.set_title('Comparison: All Methods')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('doppler_detection_methods.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'doppler_detection_methods.png'")
    plt.show()


if __name__ == "__main__":
    demonstrate_doppler_detection()
