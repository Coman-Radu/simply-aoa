import sounddevice as sd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.signal import find_peaks
import queue

# Constants for DOA calculation
SPEED_OF_SOUND = 343.0  # m/s at room temperature
MIC_SPACING = 0.042  # meters (spacing between mics in the 4x4 array)

# Audio settings
SAMPLE_RATE = 48000  # Sample rate for 16-channel array
CHANNELS = 16  # 16 microphones in 4x4 array
BUFFER_DURATION = 0.5  # seconds of audio to process
FFT_SIZE = 2048  # FFT size for GCC-PHAT

# Multi-source detection settings
MAX_SOURCES = 8  # Maximum number of sources (channels / 2)
MIN_SEPARATION = 15  # Minimum angular separation between sources (degrees)
AZIMUTH_RESOLUTION = 2  # Degrees (scanning resolution)
MIN_PEAK_HEIGHT = 0.3  # Minimum relative power for peak detection (0-1)
MIN_PROMINENCE = 0.15  # How much peak must stand out from background
SOUND_THRESHOLD = 0.002  # RMS threshold for sound detection

# GCC-PHAT settings
PAIR_SELECTION = 'all'  # 'all', 'adjacent', or 'outer'
MAX_TAU_SAMPLES = 20  # Maximum time delay in samples (limits search range)

# Queue for audio data (non-blocking)
audio_queue = queue.Queue(maxsize=10)

# Audio buffer to accumulate samples
audio_buffer = []

# Latest detection results
latest_sources = []
latest_spectrum = None
sound_active = False


def get_mic_positions_4x4():
    """
    Generate microphone positions for 4x4 array with origin at CENTER of array.

    Corrected geometry:
    Row 3 (North, Y=+63mm): MIC09, MIC08, MIC11, MIC10 → channels [8, 7, 10, 9]
    Row 2 (Y=+21mm): MIC07, MIC06, MIC13, MIC12 → channels [6, 5, 12, 11]
    Row 1 (Y=-21mm): MIC05, MIC04, MIC15, MIC14 → channels [4, 3, 14, 13]
    Row 0 (South, Y=-63mm): MIC03, MIC02, MIC01, MIC16 → channels [2, 1, 0, 15]

    X positions (West to East): -63mm, -21mm, +21mm, +63mm

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


def get_microphone_pairs(pair_selection='all'):
    """
    Get microphone pairs for GCC-PHAT processing.

    Args:
        pair_selection: 'all', 'adjacent', or 'outer'

    Returns:
        List of tuples (ch1, ch2) representing microphone pairs
    """
    if pair_selection == 'all':
        # Use all unique pairs
        pairs = []
        for i in range(CHANNELS):
            for j in range(i + 1, CHANNELS):
                pairs.append((i, j))
        return pairs

    elif pair_selection == 'adjacent':
        # Use only adjacent microphones (in rows and columns)
        pairs = []
        # Row pairs
        rows = [[2, 1, 0, 15], [4, 3, 14, 13], [6, 5, 12, 11], [8, 7, 10, 9]]
        for row in rows:
            for i in range(len(row) - 1):
                pairs.append((row[i], row[i+1]))
        # Column pairs
        columns = [[2, 4, 6, 8], [1, 3, 5, 7], [0, 14, 12, 10], [15, 13, 11, 9]]
        for col in columns:
            for i in range(len(col) - 1):
                pairs.append((col[i], col[i+1]))
        return pairs

    elif pair_selection == 'outer':
        # Use only outer perimeter microphones
        outer_mics = [2, 1, 0, 15, 13, 11, 9, 8, 7, 6, 4]  # Perimeter in order
        pairs = []
        for i in range(len(outer_mics)):
            for j in range(i + 1, len(outer_mics)):
                pairs.append((outer_mics[i], outer_mics[j]))
        return pairs

    else:
        raise ValueError(f"Unknown pair_selection: {pair_selection}")


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


def gcc_phat(sig1, sig2, fs=SAMPLE_RATE, max_tau=None):
    """
    Compute GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
    to estimate time delay between two signals.

    Args:
        sig1: First signal
        sig2: Second signal
        fs: Sample rate
        max_tau: Maximum time delay in samples (for limiting search range)

    Returns:
        tau: Time delay in samples (sig2 relative to sig1)
        gcc: GCC-PHAT function values
    """
    # Pad signals to FFT size
    n = FFT_SIZE

    # FFT of both signals
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)

    # Cross-power spectrum
    R = SIG1 * np.conj(SIG2)

    # PHAT weighting: normalize by magnitude
    # This whitens the cross-spectrum, emphasizing phase information
    R_phat = R / (np.abs(R) + 1e-10)

    # Inverse FFT to get GCC
    gcc = np.fft.irfft(R_phat, n=n)

    # Shift to center (make symmetric around zero)
    gcc = np.fft.fftshift(gcc)

    # Limit search range if specified
    if max_tau is not None:
        center = len(gcc) // 2
        gcc_limited = np.zeros_like(gcc)
        start = max(0, center - max_tau)
        end = min(len(gcc), center + max_tau + 1)
        gcc_limited[start:end] = gcc[start:end]
        gcc = gcc_limited

    # Find peak
    peak_idx = np.argmax(gcc)

    # Convert to time delay (in samples)
    tau = peak_idx - len(gcc) // 2

    return tau, gcc


def detect_multiple_sources_gcc_phat(audio_data, num_sources=MAX_SOURCES,
                                      min_separation=MIN_SEPARATION,
                                      azimuth_resolution=AZIMUTH_RESOLUTION,
                                      pair_selection=PAIR_SELECTION):
    """
    Detect multiple sound sources using GCC-PHAT.

    GCC-PHAT estimates time delays between microphone pairs, which are then
    used to compute a spatial likelihood function across all azimuth directions.

    Args:
        audio_data: Audio data array [samples, channels]
        num_sources: Maximum number of sources to detect
        min_separation: Minimum angular separation between sources (degrees)
        azimuth_resolution: Angular resolution for scanning (degrees)
        pair_selection: Microphone pair selection strategy

    Returns:
        sources: List of dictionaries with 'azimuth', 'power', 'confidence', 'id'
        spectrum_data: Tuple of (azimuth_scan, gcc_spectrum) for visualization
        sound_detected: Boolean indicating if sound was detected
    """
    # Check if sound is detected
    avg_signal = np.mean(audio_data, axis=1)
    sound_detected = detect_sound(avg_signal, threshold=SOUND_THRESHOLD)

    if not sound_detected:
        return [], None, False

    # Get mic positions and pairs
    mic_positions = get_mic_positions_4x4()
    pairs = get_microphone_pairs(pair_selection)

    # Estimate time delays for all pairs using GCC-PHAT
    measured_delays = {}
    for ch1, ch2 in pairs:
        sig1 = audio_data[:, ch1]
        sig2 = audio_data[:, ch2]

        tau, _ = gcc_phat(sig1, sig2, fs=SAMPLE_RATE, max_tau=MAX_TAU_SAMPLES)

        measured_delays[(ch1, ch2)] = tau

    # Azimuth scan range
    azimuth_scan = np.arange(0, 360, azimuth_resolution)
    gcc_spectrum = np.zeros(len(azimuth_scan))

    # For each azimuth, compute expected delays and compare with measurements
    for idx, azimuth in enumerate(azimuth_scan):
        # Convert azimuth to radians (0° = North, 90° = East)
        theta_rad = np.radians(90 - azimuth)

        # Source direction unit vector
        source_direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

        # Compute spatial likelihood by comparing expected and measured delays
        likelihood = 0.0

        for (ch1, ch2), measured_tau in measured_delays.items():
            # Get positions
            pos1 = np.array(mic_positions[ch1])
            pos2 = np.array(mic_positions[ch2])

            # Expected time delay for this direction
            # Distance difference = (pos2 - pos1) · source_direction
            distance_diff = np.dot(pos2 - pos1, source_direction)
            expected_tau = -distance_diff * SAMPLE_RATE / SPEED_OF_SOUND

            # Compare with measured delay
            # Use Gaussian-like weighting
            error = measured_tau - expected_tau
            weight = np.exp(-0.5 * (error / 3.0) ** 2)  # Sigma = 3 samples

            likelihood += weight

        gcc_spectrum[idx] = likelihood

    # Normalize spectrum
    gcc_spectrum = gcc_spectrum / (np.max(gcc_spectrum) + 1e-10)

    # Find peaks
    peaks, _ = find_peaks(
        gcc_spectrum,
        height=MIN_PEAK_HEIGHT,
        distance=int(min_separation / azimuth_resolution),
        prominence=MIN_PROMINENCE
    )

    # Sort peaks by power
    if len(peaks) > 0:
        peak_powers = gcc_spectrum[peaks]
        sorted_indices = np.argsort(peak_powers)[::-1]
        peaks = peaks[sorted_indices]
        peak_powers = peak_powers[sorted_indices]

        # Limit to num_sources
        peaks = peaks[:num_sources]
        peak_powers = peak_powers[:num_sources]

        # Create source list
        sources = []
        for peak_idx, power in zip(peaks, peak_powers):
            azimuth = azimuth_scan[peak_idx]

            sources.append({
                'azimuth': float(azimuth),
                'elevation': 0.0,  # Horizontal plane only
                'power': float(power),
                'confidence': float(power),
                'id': len(sources)
            })

        return sources, (azimuth_scan, gcc_spectrum), sound_detected
    else:
        return [], (azimuth_scan, gcc_spectrum), sound_detected


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
    global latest_sources, latest_spectrum, sound_active, audio_buffer

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

            # Detect multiple sources using GCC-PHAT
            sources, spectrum_data, sound_detected = detect_multiple_sources_gcc_phat(
                audio_data,
                num_sources=MAX_SOURCES,
                min_separation=MIN_SEPARATION,
                azimuth_resolution=AZIMUTH_RESOLUTION,
                pair_selection=PAIR_SELECTION
            )

            if sound_detected and len(sources) > 0:
                latest_sources = sources
                latest_spectrum = spectrum_data
                sound_active = True

                # Print detected sources
                if frame % 5 == 0:  # Print every 5 frames
                    sources_info = ", ".join([f"S{i+1}: Az={s['azimuth']:.1f}° (P:{s['power']:.2f})"
                                             for i, s in enumerate(sources)])
                    print(f">>> Detected {len(sources)} source(s): {sources_info}")
            else:
                sound_active = sound_detected
                if sound_detected and len(sources) == 0:
                    if frame % 10 == 0:
                        print(">>> Sound detected but no clear sources identified")

    # Clear previous plots
    ax1.clear()
    ax2.clear()

    # ===== Plot 1: Polar plot for DOA =====
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_title('Multi-Source DOA\n(GCC-PHAT)', fontsize=12, weight='bold')

    if sound_active and len(latest_sources) > 0:
        # Define colors for different sources
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow', 'magenta']

        # Plot each detected source
        for idx, source in enumerate(latest_sources):
            azimuth_rad = np.radians(source['azimuth'])
            color = colors[idx % len(colors)]

            # Arrow length based on confidence
            arrow_length = 0.6 + 0.3 * source['confidence']

            ax1.arrow(azimuth_rad, 0, 0, arrow_length,
                     head_width=0.25, head_length=0.1,
                     fc=color, ec=color, linewidth=2.5, alpha=0.8)

            # Label with source number and azimuth
            ax1.text(azimuth_rad, arrow_length + 0.15,
                    f"S{idx+1}\n{source['azimuth']:.1f}°",
                    ha='center', va='bottom', fontsize=9, color=color, weight='bold')

        # Add grid
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

    else:
        ax1.text(0, 0, 'No sound\ndetected',
                ha='center', va='center', fontsize=12, color='gray')
        ax1.set_ylim(0, 1)

    # ===== Plot 2: 3D Representation =====
    # Always plot microphone array in XY plane
    mic_positions = get_mic_positions_4x4()
    for ch, pos in mic_positions.items():
        ax2.scatter(pos[0], pos[1], 0, c='blue', s=30, alpha=0.5)

    if sound_active and len(latest_sources) > 0:
        # Define colors for different sources
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow', 'magenta']

        # Plot each detected source in horizontal plane (Z=0)
        for idx, source in enumerate(latest_sources):
            azimuth_rad = np.radians(source['azimuth'])
            color = colors[idx % len(colors)]

            # Unit vector in horizontal plane (elevation = 0)
            # Azimuth: 0° = North (+Y), 90° = East (+X)
            x = np.sin(azimuth_rad)
            y = np.cos(azimuth_rad)
            z = 0

            # Scale arrow based on confidence
            scale = 0.08 + 0.04 * source['confidence']

            # Draw arrow from origin to direction
            ax2.quiver(0, 0, 0, x*scale, y*scale, z,
                      color=color, arrow_length_ratio=0.3, linewidth=2.5, alpha=0.8)

        ax2.set_xlabel('X (East) [m]', fontsize=9)
        ax2.set_ylabel('Y (North) [m]', fontsize=9)
        ax2.set_zlabel('Z (Up) [m]', fontsize=9)
        ax2.set_title(f'3D DOA - {len(latest_sources)} Source(s)\n(Horizontal Plane)',
                     fontsize=11, weight='bold')

        ax2.set_xlim(-0.1, 0.1)
        ax2.set_ylim(-0.1, 0.1)
        ax2.set_zlim(0, 0.15)

    else:
        ax2.text(0, 0, 0.05, 'No sound detected',
                ha='center', va='center', fontsize=12, color='gray')
        ax2.set_xlabel('X (East) [m]', fontsize=9)
        ax2.set_ylabel('Y (North) [m]', fontsize=9)
        ax2.set_zlabel('Z (Up) [m]', fontsize=9)
        ax2.set_xlim(-0.1, 0.1)
        ax2.set_ylim(-0.1, 0.1)
        ax2.set_zlim(0, 0.15)


if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-SOURCE GCC-PHAT - 4x4 MICROPHONE ARRAY")
    print("=" * 80)
    print(f"Array configuration: 16 microphones in 4x4 grid")
    print(f"Microphone spacing: {MIC_SPACING*1000:.1f} mm")
    print(f"Origin: CENTER of array (0,0)")
    print(f"Detection method: GCC-PHAT (Generalized Cross-Correlation)")
    print(f"Maximum sources detected: {MAX_SOURCES} (channels / 2)")
    print(f"Pair selection: {PAIR_SELECTION}")

    pairs = get_microphone_pairs(PAIR_SELECTION)
    print(f"Number of microphone pairs: {len(pairs)}")

    print(f"Minimum separation: {MIN_SEPARATION}°")
    print(f"Azimuth resolution: {AZIMUTH_RESOLUTION}°")
    print(f"Peak detection threshold: {MIN_PEAK_HEIGHT}")
    print(f"Sound threshold: {SOUND_THRESHOLD}")
    print(f"FFT size: {FFT_SIZE}")
    print(f"Max time delay: {MAX_TAU_SAMPLES} samples ({MAX_TAU_SAMPLES/SAMPLE_RATE*1000:.2f} ms)")
    print("Note: All sources detected in horizontal plane (elevation = 0°)")
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

    print(f"\nStarting real-time GCC-PHAT-based multi-source detection...")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS}")
    print(f"Buffer duration: {BUFFER_DURATION} seconds")

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
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='polar')
    ax2 = fig.add_subplot(122, projection='3d')

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
