"""
Shared utilities for beamforming DOA estimation experiments.

Provides microphone array geometry, signal simulation (using a real WAV file
as source), delay-and-sum beamforming, and SNR-filtered peak detection.
"""

import numpy as np
from scipy.signal import find_peaks, resample, butter, sosfilt
from scipy.io import wavfile
import os

# Constants
SPEED_OF_SOUND = 343.0  # m/s
MIC_SPACING = 0.042  # meters
SAMPLE_RATE = 48000
CHANNELS = 16

# Source parameters
SOURCE_DISTANCE = 10.0  # meters
SOURCE_AZIMUTH_INITIAL = 30.0  # degrees (0deg = North, clockwise)
MULTI_SOURCE_INCREMENT = 45.0  # degrees between sources
NUM_SOURCES = 3
SOURCE_FREQUENCY = 3000.0  # Hz (used only as label / fallback)

# Moving source parameters
SOURCE_VELOCITY = 5.0  # m/s tangential speed
SIMULATION_DURATION = 4.0  # seconds
TIME_WINDOW = 0.2  # seconds per beamforming frame

# Beamforming parameters
AZIMUTH_RESOLUTION = 2  # degrees
MIN_PEAK_HEIGHT = 0.15
MIN_PROMINENCE = 0.08
MIN_SEPARATION = 10  # degrees

# SNR parameters
SNR_DB = 10  # dB
SNR_THRESHOLD_DB = 6.0
NOISE_PERCENTILE = 25

# WAV source file
WAV_FILENAME = "Binary_Drone_Audio_yes_drone_B_S2_D1_067-bebop_000_.wav"


def _get_wav_path():
    """Return absolute path to the WAV source file."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), WAV_FILENAME)


def load_source_wav(duration=None):
    """
    Load the drone WAV file and resample to SAMPLE_RATE.

    Args:
        duration: Desired duration in seconds. If the WAV is shorter it will
                  be looped; if longer it will be truncated.

    Returns:
        t: Time vector
        signal: Normalized mono signal at SAMPLE_RATE
    """
    sr, data = wavfile.read(_get_wav_path())
    # Convert to float
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    else:
        data = data.astype(np.float64)

    # If stereo, take first channel
    if data.ndim > 1:
        data = data[:, 0]

    # Resample to SAMPLE_RATE if needed
    if sr != SAMPLE_RATE:
        num_target = int(len(data) * SAMPLE_RATE / sr)
        data = resample(data, num_target)

    # Handle duration: loop or truncate
    if duration is not None:
        target_samples = int(duration * SAMPLE_RATE)
        if len(data) < target_samples:
            repeats = target_samples // len(data) + 1
            data = np.tile(data, repeats)[:target_samples]
        else:
            data = data[:target_samples]

    # Bandpass filter to keep frequencies where the array has spatial
    # resolution.  For 42mm spacing the spatial Nyquist is ~4 kHz;
    # we keep 1-7 kHz to get a usable beamforming response.
    nyq = SAMPLE_RATE / 2
    low, high = 1000.0 / nyq, 7000.0 / nyq
    if high < 1.0:
        sos = butter(4, [low, high], btype='band', output='sos')
        data = sosfilt(sos, data)

    # Normalize
    data /= np.max(np.abs(data)) + 1e-10

    t = np.arange(len(data)) / SAMPLE_RATE
    return t, data


def get_mic_positions_4x4():
    """Generate 4x4 mic array positions centered at origin."""
    outer_spacing = MIC_SPACING
    center_offset_x = 1.5 * outer_spacing
    center_offset_y = 1.5 * outer_spacing
    inner_half_spacing = outer_spacing / 2

    mic_positions = {
        1: (0 - center_offset_x, 0 - center_offset_y),
        3: (outer_spacing - center_offset_x, 0 - center_offset_y),
        5: (2 * outer_spacing - center_offset_x, 0 - center_offset_y),
        7: (3 * outer_spacing - center_offset_x, 0 - center_offset_y),
        6: (3 * outer_spacing - center_offset_x, outer_spacing - center_offset_y),
        9: (3 * outer_spacing - center_offset_x, 2 * outer_spacing - center_offset_y),
        8: (3 * outer_spacing - center_offset_x, 3 * outer_spacing - center_offset_y),
        10: (2 * outer_spacing - center_offset_x, 3 * outer_spacing - center_offset_y),
        12: (outer_spacing - center_offset_x, 3 * outer_spacing - center_offset_y),
        14: (0 - center_offset_x, 3 * outer_spacing - center_offset_y),
        15: (0 - center_offset_x, 2 * outer_spacing - center_offset_y),
        0: (0 - center_offset_x, outer_spacing - center_offset_y),
        2: (-inner_half_spacing, -inner_half_spacing),
        4: (inner_half_spacing, -inner_half_spacing),
        11: (inner_half_spacing, inner_half_spacing),
        13: (-inner_half_spacing, inner_half_spacing)
    }
    return mic_positions


def simulate_array_signals(source_signal, source_azimuth_deg, source_distance,
                           snr_db=SNR_DB):
    """
    Simulate mic array signals for a far-field point source.

    Args:
        source_signal: Mono source waveform
        source_azimuth_deg: Azimuth in degrees (0deg=North, clockwise)
        source_distance: Distance in meters
        snr_db: Signal-to-noise ratio in dB

    Returns:
        mic_signals: [samples, channels]
    """
    mic_positions = get_mic_positions_4x4()
    num_samples = len(source_signal)

    theta_rad = np.radians(90 - source_azimuth_deg)
    source_direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    signal_power = np.mean(source_signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)

    mic_signals = np.zeros((num_samples, CHANNELS))

    sample_indices = np.arange(num_samples, dtype=np.float64)

    for ch in range(CHANNELS):
        pos = np.array(mic_positions[ch])
        distance_proj = np.dot(pos, source_direction)
        time_delay = distance_proj / SPEED_OF_SOUND
        delay_samples = time_delay * SAMPLE_RATE  # fractional

        # Fractional delay via linear interpolation
        shifted_indices = sample_indices + delay_samples
        delayed = np.interp(shifted_indices, sample_indices, source_signal,
                            left=0.0, right=0.0)

        mic_distance = source_distance + distance_proj
        attenuation = source_distance / mic_distance

        noise = np.random.randn(num_samples) * noise_std
        mic_signals[:, ch] = delayed * attenuation + noise

    return mic_signals


def _steering_vector(azimuth_deg, freqs, mic_positions):
    """Compute steering vector for given azimuth and frequencies."""
    theta_rad = np.radians(90 - azimuth_deg)
    direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    positions = np.array([mic_positions[ch] for ch in range(CHANNELS)])
    delays = positions @ direction / SPEED_OF_SOUND  # [channels]
    # steering vector: [freqs, channels]
    phase = 2j * np.pi * freqs[:, None] * delays[None, :]
    return np.exp(phase)


def beamforming_scan(audio_data, method='mvdr'):
    """
    Beamforming scan across all azimuths.

    Args:
        audio_data: [samples, channels]
        method: 'das' for delay-and-sum, 'mvdr' for Capon/MVDR
    """
    mic_positions = get_mic_positions_4x4()
    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
    azimuth_scan = np.arange(0, 360, AZIMUTH_RESOLUTION)
    num_samples = len(audio_data)

    if method == 'das':
        # Classic delay-and-sum
        power_spectrum = np.zeros(len(azimuth_scan))
        sample_indices = np.arange(num_samples, dtype=np.float64)

        for idx, azimuth in enumerate(azimuth_scan):
            theta_rad = np.radians(90 - azimuth)
            source_direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

            delayed_signals = []
            for ch in range(CHANNELS):
                pos = np.array(mic_positions[ch])
                distance_proj = np.dot(pos, source_direction)
                time_delay = distance_proj / SPEED_OF_SOUND
                delay_samples = time_delay * SAMPLE_RATE

                shifted_indices = sample_indices - delay_samples
                delayed = np.interp(shifted_indices, sample_indices,
                                    audio_data[:, ch], left=0.0, right=0.0)
                delayed_signals.append(delayed)

            beamformed = np.mean(delayed_signals, axis=0)
            power_spectrum[idx] = np.mean(beamformed ** 2)
    else:
        # MVDR (Capon) beamforming — per-frequency-bin narrowband
        nfft = min(1024, num_samples)
        # Use multiple overlapping segments for better covariance estimate
        hop = nfft // 2
        n_segments = max(1, (num_samples - nfft) // hop + 1)

        X_all = np.fft.rfft(audio_data[:nfft, :], n=nfft, axis=0)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / SAMPLE_RATE)

        # Use full bandwidth (500 Hz to Nyquist) — spatial aliasing
        # artifacts above c/(2d) ≈ 4 kHz are intentionally visible
        freq_mask = freqs >= 500
        freq_indices = np.where(freq_mask)[0]
        freqs_used = freqs[freq_indices]

        # Accumulate per-frequency cross-spectral matrices
        R_per_freq = np.zeros((len(freq_indices), CHANNELS, CHANNELS),
                              dtype=complex)
        for s in range(n_segments):
            start = s * hop
            segment = audio_data[start:start + nfft, :]
            if len(segment) < nfft:
                break
            X_seg = np.fft.rfft(segment, n=nfft, axis=0)
            for fi_out, fi in enumerate(freq_indices):
                x = X_seg[fi, :]  # [channels]
                R_per_freq[fi_out] += np.outer(x, np.conj(x))
        R_per_freq /= n_segments

        # Diagonal loading per frequency bin
        for fi_out in range(len(freq_indices)):
            reg = np.trace(R_per_freq[fi_out]).real / CHANNELS * 0.01
            R_per_freq[fi_out] += reg * np.eye(CHANNELS)

        # Precompute inverses
        R_inv_per_freq = np.linalg.inv(R_per_freq)

        power_spectrum = np.zeros(len(azimuth_scan))
        for idx, azimuth in enumerate(azimuth_scan):
            a = _steering_vector(azimuth, freqs_used,
                                 mic_positions)  # [freqs, ch]
            p_sum = 0.0
            for fi in range(len(freq_indices)):
                av = a[fi, :]
                denom = np.real(av.conj() @ R_inv_per_freq[fi] @ av)
                p_sum += 1.0 / (denom + 1e-20)
            power_spectrum[idx] = p_sum / len(freq_indices)

    power_spectrum = power_spectrum / (np.max(power_spectrum) + 1e-10)
    return azimuth_scan, power_spectrum


def detect_peaks(azimuth_scan, power_spectrum):
    """Detect peaks with SNR filtering."""
    azimuth_resolution = azimuth_scan[1] - azimuth_scan[0]
    noise_floor = np.percentile(power_spectrum, NOISE_PERCENTILE)

    peak_indices, _ = find_peaks(
        power_spectrum,
        height=MIN_PEAK_HEIGHT,
        distance=int(MIN_SEPARATION / azimuth_resolution),
        prominence=MIN_PROMINENCE
    )

    peaks = []
    for idx in peak_indices:
        peak_power = power_spectrum[idx]
        if noise_floor > 0:
            snr_db = 10 * np.log10(peak_power / noise_floor)
        else:
            snr_db = float('inf')
        if snr_db >= SNR_THRESHOLD_DB:
            peaks.append({
                'azimuth': azimuth_scan[idx],
                'power': peak_power,
                'snr_db': snr_db
            })

    peaks = sorted(peaks, key=lambda x: x['power'], reverse=True)
    return peaks, noise_floor
