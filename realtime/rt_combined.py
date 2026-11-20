import sounddevice as sd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.signal import find_peaks, stft
from scipy.cluster.hierarchy import fclusterdata
import queue

# Constants for DOA calculation
SPEED_OF_SOUND = 343.0  # m/s at room temperature
MIC_SPACING = 0.042  # meters (spacing between mics in the 4x4 array)

# Audio settings
SAMPLE_RATE = 48000  # Sample rate for 16-channel array
CHANNELS = 16  # 16 microphones in 4x4 array
BUFFER_DURATION = 0.5  # seconds of audio to process

# STFT settings
STFT_NPERSEG = 512  # STFT window length
STFT_NOVERLAP = 256  # STFT overlap
STFT_NFFT = 512  # STFT FFT length
ENERGY_THRESHOLD = 0.01  # Minimum energy in TF bin to process

# SRP-PHAT settings
AZIMUTH_RESOLUTION = 2  # Degrees (scanning resolution)
SRP_ACCUMULATION_WEIGHT = 'energy'  # 'energy' or 'uniform'

# Peak detection and clustering settings
MIN_SEPARATION = 15  # Minimum angular separation between sources (degrees)
MIN_PEAK_HEIGHT = 0.3  # Minimum relative power for peak detection (0-1)
MIN_PROMINENCE = 0.15  # How much peak must stand out from background
CLUSTERING_THRESHOLD = 20  # Maximum distance (degrees) for clustering peaks
SOUND_THRESHOLD = 0.002  # RMS threshold for sound detection

# Eigenanalysis settings
USE_MDL = True  # Use MDL criterion (else use AIC)
MAX_SOURCES_EIGEN = 8  # Maximum sources to consider in MDL/AIC
EIGENVALUE_THRESHOLD = 0.01  # Minimum eigenvalue to consider as signal

# MUSIC settings (for refinement)
USE_MUSIC_REFINEMENT = True  # Apply MUSIC after eigen-analysis
MUSIC_SPATIAL_SMOOTHING = True  # Use spatial smoothing in MUSIC

# Queue for audio data (non-blocking)
audio_queue = queue.Queue(maxsize=10)

# Audio buffer to accumulate samples
audio_buffer = []

# Latest detection results
latest_sources = []
latest_srp_map = None
latest_music_spectrum = None
sound_active = False
estimated_num_sources = 0


def get_mic_positions_4x4():
    """
    Generate microphone positions for 4x4 array with origin at CENTER of array.

    Returns:
        Dictionary mapping channel numbers to (x, y) positions in meters
    """
    spacing = MIC_SPACING

    # X positions (West to East)
    x_positions = [-1.5 * spacing, -0.5 * spacing, 0.5 * spacing, 1.5 * spacing]

    # Y positions (South to North)
    y_positions = [-1.5 * spacing, -0.5 * spacing, 0.5 * spacing, 1.5 * spacing]

    # Build position dictionary for each channel
    mic_positions = {}

    # Row 0 (South, Y=-63mm): channels [2, 1, 0, 15]
    mic_positions[2] = (x_positions[0], y_positions[0])  # MIC03
    mic_positions[1] = (x_positions[1], y_positions[0])  # MIC02
    mic_positions[0] = (x_positions[2], y_positions[0])  # MIC01
    mic_positions[15] = (x_positions[3], y_positions[0])  # MIC16

    # Row 1 (Y=-21mm): channels [4, 3, 14, 13]
    mic_positions[4] = (x_positions[0], y_positions[1])  # MIC05
    mic_positions[3] = (x_positions[1], y_positions[1])  # MIC04
    mic_positions[14] = (x_positions[2], y_positions[1])  # MIC15
    mic_positions[13] = (x_positions[3], y_positions[1])  # MIC14

    # Row 2 (Y=+21mm): channels [6, 5, 12, 11]
    mic_positions[6] = (x_positions[0], y_positions[2])  # MIC07
    mic_positions[5] = (x_positions[1], y_positions[2])  # MIC06
    mic_positions[12] = (x_positions[2], y_positions[2])  # MIC13
    mic_positions[11] = (x_positions[3], y_positions[2])  # MIC12

    # Row 3 (North, Y=+63mm): channels [8, 7, 10, 9]
    mic_positions[8] = (x_positions[0], y_positions[3])  # MIC09
    mic_positions[7] = (x_positions[1], y_positions[3])  # MIC08
    mic_positions[10] = (x_positions[2], y_positions[3])  # MIC11
    mic_positions[9] = (x_positions[3], y_positions[3])  # MIC10

    return mic_positions


def detect_sound(signal, threshold=SOUND_THRESHOLD):
    """
    Detect if there's a sound event in the signal.

    Args:
        signal: Audio signal
        threshold: RMS threshold for sound detection

    Returns:
        bool: True if sound detected, False otherwise
    """
    rms = np.sqrt(np.mean(signal**2))
    return rms > threshold


def compute_steering_vector(azimuth, frequency, mic_positions):
    """
    Compute the steering vector for a given azimuth direction and frequency.

    Args:
        azimuth: Azimuth angle in degrees (0° = North, clockwise)
        frequency: Frequency in Hz
        mic_positions: Dictionary of microphone positions

    Returns:
        Steering vector (complex array of shape [CHANNELS])
    """
    # Convert azimuth to radians (0° = North, 90° = East)
    theta_rad = np.radians(90 - azimuth)

    # Source direction unit vector
    source_direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    # Calculate phase delays for each microphone
    steering_vector = np.zeros(CHANNELS, dtype=complex)

    for ch in range(CHANNELS):
        pos = np.array(mic_positions[ch])

        # Time delay for this mic relative to origin
        distance_proj = np.dot(pos, source_direction)
        time_delay = -distance_proj / SPEED_OF_SOUND

        # Convert time delay to phase at this frequency
        phase = 2 * np.pi * frequency * time_delay

        steering_vector[ch] = np.exp(1j * phase)

    return steering_vector


def compute_srp_phat_tf(stft_data, freqs, mic_positions, azimuth_resolution=AZIMUTH_RESOLUTION):
    """
    Compute SRP-PHAT map by accumulating across time-frequency bins.

    Args:
        stft_data: STFT data [channels, freq_bins, time_frames]
        freqs: Frequency bins
        mic_positions: Dictionary of microphone positions
        azimuth_resolution: Angular resolution in degrees

    Returns:
        srp_map: Accumulated SRP-PHAT map across all azimuths
        azimuth_scan: Azimuth angles scanned
    """
    num_freq_bins = stft_data.shape[1]
    num_time_frames = stft_data.shape[2]

    # Azimuth scan range
    azimuth_scan = np.arange(0, 360, azimuth_resolution)
    srp_map = np.zeros(len(azimuth_scan))

    # Process each time-frequency bin
    for t_idx in range(num_time_frames):
        for f_idx in range(num_freq_bins):
            # Skip low frequencies and DC
            if freqs[f_idx] < 500:
                continue

            # Extract signals at this TF bin
            X = stft_data[:, f_idx, t_idx]  # [channels]

            # Compute energy in this bin
            energy = np.mean(np.abs(X)**2)

            # Skip if energy too low
            if energy < ENERGY_THRESHOLD:
                continue

            # Compute cross-spectrum with PHAT normalization
            # R = X * X^H, then PHAT normalize
            R = np.outer(X, np.conj(X))

            # SRP-PHAT for this TF bin
            for az_idx, azimuth in enumerate(azimuth_scan):
                # Get steering vector for this direction and frequency
                a = compute_steering_vector(azimuth, freqs[f_idx], mic_positions)

                # SRP-PHAT: a^H * R_phat * a
                # For PHAT, we normalize each element of R
                R_phat = R / (np.abs(R) + 1e-10)

                # Compute power
                power = np.real(np.dot(np.conj(a), np.dot(R_phat, a)))

                # Add to accumulator with weighting
                if SRP_ACCUMULATION_WEIGHT == 'energy':
                    srp_map[az_idx] += power * energy
                else:
                    srp_map[az_idx] += power

    # Normalize
    srp_map = srp_map / (np.max(srp_map) + 1e-10)

    return srp_map, azimuth_scan


def estimate_num_sources_mdl_aic(covariance_matrix, num_samples):
    """
    Estimate number of sources using MDL or AIC criterion.

    Args:
        covariance_matrix: Spatial covariance matrix [CHANNELS, CHANNELS]
        num_samples: Number of samples used to compute covariance

    Returns:
        num_sources: Estimated number of sources
        eigenvalues: Eigenvalues (sorted descending)
    """
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]

    # Filter out very small eigenvalues
    eigenvalues = eigenvalues[eigenvalues > EIGENVALUE_THRESHOLD * eigenvalues[0]]

    M = len(eigenvalues)  # Number of sensors (effective)
    N = num_samples  # Number of snapshots

    min_criterion = np.inf
    best_k = 0

    # Test for k = 0 to min(M-1, MAX_SOURCES_EIGEN)
    for k in range(min(M, MAX_SOURCES_EIGEN + 1)):
        if k >= M:
            break

        # Geometric mean of noise eigenvalues
        if k < M:
            noise_eigenvalues = eigenvalues[k:]
            geometric_mean = np.exp(np.mean(np.log(noise_eigenvalues + 1e-10)))
            arithmetic_mean = np.mean(noise_eigenvalues)

            # Likelihood ratio
            if arithmetic_mean > 0 and geometric_mean > 0:
                log_likelihood = -N * (M - k) * np.log(geometric_mean / arithmetic_mean)
            else:
                continue
        else:
            log_likelihood = 0

        # Penalty term
        if USE_MDL:
            # MDL criterion
            penalty = 0.5 * k * (2 * M - k) * np.log(N)
        else:
            # AIC criterion
            penalty = k * (2 * M - k)

        criterion = -log_likelihood + penalty

        if criterion < min_criterion:
            min_criterion = criterion
            best_k = k

    return best_k, eigenvalues


def detect_peaks_and_cluster(srp_map, azimuth_scan):
    """
    Detect peaks in SRP map and cluster them.

    Args:
        srp_map: SRP-PHAT map
        azimuth_scan: Azimuth angles

    Returns:
        clustered_peaks: List of (azimuth, power) tuples for each cluster
    """
    # Find peaks
    peaks, properties = find_peaks(
        srp_map,
        height=MIN_PEAK_HEIGHT,
        distance=int(MIN_SEPARATION / AZIMUTH_RESOLUTION),
        prominence=MIN_PROMINENCE
    )

    if len(peaks) == 0:
        return []

    # Get peak azimuths and powers
    peak_azimuths = azimuth_scan[peaks]
    peak_powers = srp_map[peaks]

    # Cluster peaks (handle angular wraparound)
    if len(peaks) > 1:
        # Convert to Cartesian for clustering
        peak_coords = np.column_stack([
            np.cos(np.radians(peak_azimuths)),
            np.sin(np.radians(peak_azimuths))
        ])

        # Cluster
        try:
            clusters = fclusterdata(peak_coords,
                                   t=np.sin(np.radians(CLUSTERING_THRESHOLD/2)) * 2,
                                   criterion='distance',
                                   method='single')

            # Average peaks in each cluster
            clustered_peaks = []
            for cluster_id in np.unique(clusters):
                cluster_mask = clusters == cluster_id
                cluster_azimuths = peak_azimuths[cluster_mask]
                cluster_powers = peak_powers[cluster_mask]

                # Weighted average of azimuth
                avg_azimuth = np.degrees(np.arctan2(
                    np.sum(cluster_powers * np.sin(np.radians(cluster_azimuths))),
                    np.sum(cluster_powers * np.cos(np.radians(cluster_azimuths)))
                )) % 360

                avg_power = np.mean(cluster_powers)

                clustered_peaks.append((avg_azimuth, avg_power))
        except:
            # If clustering fails, just use raw peaks
            clustered_peaks = list(zip(peak_azimuths, peak_powers))
    else:
        clustered_peaks = list(zip(peak_azimuths, peak_powers))

    # Sort by power
    clustered_peaks.sort(key=lambda x: x[1], reverse=True)

    return clustered_peaks


def refine_doa_with_music(audio_data, num_sources, initial_azimuths, mic_positions):
    """
    Refine DOA estimates using MUSIC algorithm.

    Args:
        audio_data: Audio data [samples, channels]
        num_sources: Number of sources (from eigen-analysis)
        initial_azimuths: Initial azimuth estimates from SRP
        mic_positions: Microphone positions

    Returns:
        refined_azimuths: Refined azimuth estimates
        music_spectrum: MUSIC spectrum for visualization
    """
    # Compute covariance matrix
    R = np.dot(audio_data.T, np.conj(audio_data.T).T) / len(audio_data)

    # Spatial smoothing (if enabled)
    if MUSIC_SPATIAL_SMOOTHING and CHANNELS > num_sources:
        # Simple spatial smoothing: average over subarrays
        subarray_size = CHANNELS - num_sources + 1
        R_smoothed = np.zeros_like(R)

        for i in range(subarray_size):
            # Create subarray indices
            subarray_indices = np.arange(i, i + num_sources)
            if np.max(subarray_indices) < CHANNELS:
                R_sub = R[np.ix_(subarray_indices, subarray_indices)]
                R_smoothed[np.ix_(subarray_indices, subarray_indices)] += R_sub

        R = R_smoothed / subarray_size

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R)

    # Sort eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Noise subspace
    noise_subspace = eigenvectors[:, num_sources:]

    # Compute MUSIC spectrum
    azimuth_scan = np.arange(0, 360, AZIMUTH_RESOLUTION)
    music_spectrum = np.zeros(len(azimuth_scan))

    # Use center frequency for steering vector
    center_freq = 2000  # Hz

    for idx, azimuth in enumerate(azimuth_scan):
        a = compute_steering_vector(azimuth, center_freq, mic_positions)

        # MUSIC pseudo-spectrum
        projection = np.dot(noise_subspace, np.conj(noise_subspace).T)
        denominator = np.dot(np.conj(a).T, np.dot(projection, a))

        music_spectrum[idx] = 1.0 / (np.abs(denominator) + 1e-10)

    # Normalize
    music_spectrum = music_spectrum / (np.max(music_spectrum) + 1e-10)

    # Find peaks around initial estimates
    refined_azimuths = []
    for init_az in initial_azimuths:
        # Search in a window around initial estimate
        search_window = 30  # degrees
        search_start = int((init_az - search_window) / AZIMUTH_RESOLUTION) % len(azimuth_scan)
        search_end = int((init_az + search_window) / AZIMUTH_RESOLUTION) % len(azimuth_scan)

        # Handle wraparound
        if search_start < search_end:
            window_indices = np.arange(search_start, search_end)
        else:
            window_indices = np.concatenate([
                np.arange(search_start, len(azimuth_scan)),
                np.arange(0, search_end)
            ])

        # Find peak in window
        window_spectrum = music_spectrum[window_indices]
        peak_idx = window_indices[np.argmax(window_spectrum)]

        refined_azimuths.append(azimuth_scan[peak_idx])

    return refined_azimuths, music_spectrum


def combined_source_estimation(audio_data):
    """
    Combined source estimation using SRP-PHAT, eigen-analysis, and MUSIC.

    Pipeline:
    1. Compute STFT
    2. Accumulate SRP-PHAT across TF bins
    3. Detect peaks and cluster
    4. Compute covariance and estimate num sources (MDL/AIC)
    5. Compare SRP and eigen estimates
    6. Optionally refine with MUSIC

    Args:
        audio_data: Audio data [samples, channels]

    Returns:
        sources: List of detected sources with azimuth, power, etc.
        srp_map: SRP-PHAT map
        music_spectrum: MUSIC spectrum (if refinement used)
        num_sources_eigen: Estimated number of sources from eigenanalysis
        num_sources_srp: Estimated number of sources from SRP peaks
    """
    # Check if sound is detected
    avg_signal = np.mean(audio_data, axis=1)
    sound_detected = detect_sound(avg_signal, threshold=SOUND_THRESHOLD)

    if not sound_detected:
        return [], None, None, 0, 0

    mic_positions = get_mic_positions_4x4()

    # Step 1: Compute STFT for all channels
    stft_data = []
    for ch in range(CHANNELS):
        f, t, Zxx = stft(audio_data[:, ch],
                        fs=SAMPLE_RATE,
                        nperseg=STFT_NPERSEG,
                        noverlap=STFT_NOVERLAP,
                        nfft=STFT_NFFT)
        stft_data.append(Zxx)

    stft_data = np.array(stft_data)  # [channels, freq_bins, time_frames]

    # Step 2: Compute SRP-PHAT across TF bins
    srp_map, azimuth_scan = compute_srp_phat_tf(stft_data, f, mic_positions)

    # Step 3: Detect peaks and cluster
    srp_peaks = detect_peaks_and_cluster(srp_map, azimuth_scan)
    num_sources_srp = len(srp_peaks)

    # Step 4: Compute covariance and estimate number of sources
    R = np.dot(audio_data.T, np.conj(audio_data.T).T) / len(audio_data)
    num_sources_eigen, eigenvalues = estimate_num_sources_mdl_aic(R, len(audio_data))

    # Step 5: Compare estimates and decide
    print(f"  SRP peaks: {num_sources_srp}, Eigen estimate: {num_sources_eigen}")

    if num_sources_srp == 0 and num_sources_eigen == 0:
        return [], srp_map, None, num_sources_eigen, num_sources_srp

    # Decision logic
    if num_sources_srp == num_sources_eigen:
        # Agreement: use SRP DOAs
        num_sources_final = num_sources_srp
        final_azimuths = [az for az, _ in srp_peaks[:num_sources_final]]
        source_powers = [pw for _, pw in srp_peaks[:num_sources_final]]
    else:
        # Disagreement: prefer eigen estimate
        num_sources_final = num_sources_eigen

        if num_sources_final > 0:
            # Use top SRP peaks, but trust eigen count
            final_azimuths = [az for az, _ in srp_peaks[:num_sources_final]]
            source_powers = [pw for _, pw in srp_peaks[:num_sources_final]]

            # Pad if needed
            while len(final_azimuths) < num_sources_final:
                final_azimuths.append(0.0)
                source_powers.append(0.1)
        else:
            final_azimuths = []
            source_powers = []

    # Step 6: Optional MUSIC refinement
    music_spectrum = None
    if USE_MUSIC_REFINEMENT and num_sources_final > 0:
        refined_azimuths, music_spectrum = refine_doa_with_music(
            audio_data, num_sources_final, final_azimuths, mic_positions
        )
        final_azimuths = refined_azimuths

    # Create source list
    sources = []
    for idx in range(len(final_azimuths)):
        sources.append({
            'azimuth': float(final_azimuths[idx]),
            'elevation': 0.0,
            'power': float(source_powers[idx]) if idx < len(source_powers) else 0.5,
            'confidence': float(source_powers[idx]) if idx < len(source_powers) else 0.5,
            'id': idx
        })

    return sources, srp_map, music_spectrum, num_sources_eigen, num_sources_srp


def audio_callback(indata, frames, time, status):
    """
    Callback function for sounddevice stream.
    """
    if status:
        print(f"Status: {status}")

    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        pass


def update_plot(frame):
    """
    Update function for matplotlib animation.
    """
    global latest_sources, latest_srp_map, latest_music_spectrum, sound_active, audio_buffer, estimated_num_sources

    # Get all available audio chunks from queue
    try:
        while True:
            chunk = audio_queue.get_nowait()
            audio_buffer.append(chunk)
    except queue.Empty:
        pass

    # Keep buffer size manageable
    max_buffer_samples = int(SAMPLE_RATE * 1.0)
    if len(audio_buffer) > 0:
        total_samples = sum(len(chunk) for chunk in audio_buffer)

        while total_samples > max_buffer_samples and len(audio_buffer) > 1:
            removed = audio_buffer.pop(0)
            total_samples -= len(removed)

    # Process if we have enough samples
    if len(audio_buffer) > 0:
        audio_data = np.vstack(audio_buffer)

        # Only process if we have enough samples
        if len(audio_data) >= int(BUFFER_DURATION * SAMPLE_RATE):
            # Take last BUFFER_DURATION worth of samples
            audio_data = audio_data[-int(BUFFER_DURATION * SAMPLE_RATE):]

            # Remove DC offset from each channel
            audio_data = audio_data - np.mean(audio_data, axis=0)

            # Normalize
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)

            # Combined source estimation
            sources, srp_map, music_spectrum, num_eigen, num_srp = combined_source_estimation(audio_data)

            if len(sources) > 0:
                latest_sources = sources
                latest_srp_map = srp_map
                latest_music_spectrum = music_spectrum
                sound_active = True
                estimated_num_sources = num_eigen

                # Print detected sources
                if frame % 5 == 0:
                    sources_info = ", ".join([f"S{i+1}: Az={s['azimuth']:.1f}°"
                                            for i, s in enumerate(sources)])
                    print(f">>> K={num_eigen} (SRP:{num_srp}, Eigen:{num_eigen}), Sources: {sources_info}")
            else:
                sound_active = False

    # Clear previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # ===== Plot 1: SRP-PHAT Map =====
    if latest_srp_map is not None:
        azimuth_scan = np.arange(0, 360, AZIMUTH_RESOLUTION)
        ax1.plot(azimuth_scan, latest_srp_map, 'b-', linewidth=2, label='SRP-PHAT')

        # Mark detected sources
        if len(latest_sources) > 0:
            for src in latest_sources:
                ax1.axvline(x=src['azimuth'], color='red', linestyle='--', alpha=0.7)

        ax1.set_xlabel('Azimuth (degrees)', fontsize=10)
        ax1.set_ylabel('Power', fontsize=10)
        ax1.set_title(f'SRP-PHAT Map (K={estimated_num_sources})', fontsize=12, weight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 360)
        ax1.legend()

    # ===== Plot 2: MUSIC Spectrum (if available) =====
    if latest_music_spectrum is not None:
        azimuth_scan = np.arange(0, 360, AZIMUTH_RESOLUTION)
        ax2.plot(azimuth_scan, latest_music_spectrum, 'g-', linewidth=2, label='MUSIC')

        # Mark detected sources
        if len(latest_sources) > 0:
            for src in latest_sources:
                ax2.axvline(x=src['azimuth'], color='red', linestyle='--', alpha=0.7)

        ax2.set_xlabel('Azimuth (degrees)', fontsize=10)
        ax2.set_ylabel('Pseudo-spectrum', fontsize=10)
        ax2.set_title('MUSIC Spectrum (Refined)', fontsize=12, weight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 360)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'MUSIC refinement\nnot available',
                ha='center', va='center', fontsize=12, color='gray',
                transform=ax2.transAxes)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

    # ===== Plot 3: Polar DOA Plot =====
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.set_title(f'DOA Estimates (K={estimated_num_sources})', fontsize=12, weight='bold')

    if sound_active and len(latest_sources) > 0:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow', 'magenta']

        for idx, source in enumerate(latest_sources):
            azimuth_rad = np.radians(source['azimuth'])
            color = colors[idx % len(colors)]

            arrow_length = 0.6 + 0.3 * source['confidence']

            ax3.arrow(azimuth_rad, 0, 0, arrow_length,
                     head_width=0.25, head_length=0.1,
                     fc=color, ec=color, linewidth=2.5, alpha=0.8)

            ax3.text(azimuth_rad, arrow_length + 0.15,
                    f"S{idx+1}\n{source['azimuth']:.1f}°",
                    ha='center', va='bottom', fontsize=9, color=color, weight='bold')

        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0, 0, 'No sound\ndetected',
                ha='center', va='center', fontsize=12, color='gray')
        ax3.set_ylim(0, 1)


if __name__ == "__main__":
    print("=" * 80)
    print("COMBINED SOURCE ESTIMATION: SRP-PHAT + EIGENANALYSIS + MUSIC")
    print("=" * 80)
    print(f"Array: 16 microphones in 4x4 grid")
    print(f"Microphone spacing: {MIC_SPACING*1000:.1f} mm")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Buffer duration: {BUFFER_DURATION} s")
    print()
    print("Pipeline:")
    print("  1. Compute STFT")
    print("  2. Accumulate SRP-PHAT across TF bins with energy weighting")
    print("  3. Detect and cluster peaks in SRP map")
    print("  4. Estimate number of sources using MDL/AIC on covariance eigenvalues")
    print("  5. Compare SRP and eigen estimates")
    print(f"  6. Refine DOAs using MUSIC: {'ENABLED' if USE_MUSIC_REFINEMENT else 'DISABLED'}")
    print(f"     - Spatial smoothing: {'ENABLED' if MUSIC_SPATIAL_SMOOTHING else 'DISABLED'}")
    print()
    print(f"Settings:")
    print(f"  - STFT: nperseg={STFT_NPERSEG}, noverlap={STFT_NOVERLAP}")
    print(f"  - Source count criterion: {'MDL' if USE_MDL else 'AIC'}")
    print(f"  - Energy threshold: {ENERGY_THRESHOLD}")
    print(f"  - Azimuth resolution: {AZIMUTH_RESOLUTION}°")
    print("=" * 80)
    print()

    # Find miniDSP device
    devices = sd.query_devices()
    minidsp_device = None

    for i, device in enumerate(devices):
        if 'line' in device['name'].lower():
            minidsp_device = i
            print(f"Found miniDSP device at index {i}")
            print(f"  Name: {device['name']}")
            print(f"  Max input channels: {device['max_input_channels']}")
            print(f"  Default sample rate: {device['default_samplerate']}")
            break

    if minidsp_device is None:
        print("Error: miniDSP device not found!")
        print("\nAvailable devices:")
        for i, device in enumerate(devices):
            print(f"  [{i}] {device['name']}")
        exit(1)

    print(f"\nStarting real-time combined source estimation...")

    # Start audio stream
    stream = sd.InputStream(
        device=minidsp_device,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1)
    )

    stream.start()

    # Set up matplotlib figure
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection='polar')

    # Create animation
    ani = animation.FuncAnimation(
        fig, update_plot, interval=200, cache_frame_data=False, blit=False
    )

    plt.tight_layout()

    try:
        print("\nReal-time processing active. Close the plot window to stop.")
        print("=" * 70)
        plt.show()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop()
        stream.close()
        print("Stream closed.")
