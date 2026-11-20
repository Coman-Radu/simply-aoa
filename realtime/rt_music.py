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

# Multi-source detection settings
MAX_SOURCES = 8  # Maximum number of sources (channels / 2)
MIN_SEPARATION = 15  # Minimum angular separation between sources (degrees)
AZIMUTH_RESOLUTION = 2  # Degrees (scanning resolution)
MIN_PEAK_HEIGHT = 0.3  # Minimum relative power for peak detection (0-1)
MIN_PROMINENCE = 0.15  # How much peak must stand out from background
SOUND_THRESHOLD = 0.002  # RMS threshold for sound detection

# MUSIC algorithm settings
NUM_SOURCES_MUSIC = 3  # Assumed number of sources for subspace separation (can be < MAX_SOURCES)
SMOOTHING_FACTOR = 0.99  # Exponential smoothing for covariance matrix

# Queue for audio data (non-blocking)
audio_queue = queue.Queue(maxsize=10)

# Audio buffer to accumulate samples
audio_buffer = []

# Latest detection results
latest_sources = []
latest_spectrum = None
sound_active = False

# Covariance matrix for spatial smoothing
covariance_matrix = None


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


def compute_steering_vector(azimuth, mic_positions):
    """
    Compute the steering vector for a given azimuth direction.

    Args:
        azimuth: Azimuth angle in degrees (0° = North, clockwise)
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

        # Convert time delay to phase
        # Assuming center frequency for narrowband assumption
        # For broadband, we'd need to do this for each frequency bin
        freq_center = 1000  # Hz (can be adjusted or made adaptive)
        phase = 2 * np.pi * freq_center * time_delay

        steering_vector[ch] = np.exp(1j * phase)

    # Normalize
    steering_vector = steering_vector / np.linalg.norm(steering_vector)

    return steering_vector


def detect_multiple_sources_music(audio_data, num_sources=MAX_SOURCES,
                                   num_sources_subspace=NUM_SOURCES_MUSIC,
                                   min_separation=MIN_SEPARATION,
                                   azimuth_resolution=AZIMUTH_RESOLUTION):
    """
    Detect multiple sound sources using MUSIC algorithm.

    MUSIC (Multiple Signal Classification) uses eigenvalue decomposition
    to separate signal and noise subspaces, providing higher resolution
    than conventional beamforming.

    Args:
        audio_data: Audio data array [samples, channels]
        num_sources: Maximum number of sources to detect from peaks
        num_sources_subspace: Number of signal sources for subspace separation
        min_separation: Minimum angular separation between sources (degrees)
        azimuth_resolution: Angular resolution for scanning (degrees)

    Returns:
        sources: List of dictionaries with 'azimuth', 'power', 'confidence', 'id'
        spectrum_data: Tuple of (azimuth_scan, music_spectrum) for visualization
        sound_detected: Boolean indicating if sound was detected
    """
    global covariance_matrix

    # Check if sound is detected
    avg_signal = np.mean(audio_data, axis=1)
    sound_detected = detect_sound(avg_signal, threshold=SOUND_THRESHOLD)

    if not sound_detected:
        return [], None, False

    # Get mic positions
    mic_positions = get_mic_positions_4x4()

    # Compute spatial covariance matrix
    # R = E[x(t) * x(t)^H] where x(t) is the signal vector
    # Shape: [CHANNELS, CHANNELS]
    R = np.dot(audio_data.T, np.conj(audio_data.T).T) / len(audio_data)

    # Apply exponential smoothing if we have previous covariance
    if covariance_matrix is not None:
        R = SMOOTHING_FACTOR * covariance_matrix + (1 - SMOOTHING_FACTOR) * R

    covariance_matrix = R

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R)

    # Sort eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Separate signal and noise subspaces
    # Signal subspace: first num_sources_subspace eigenvectors
    # Noise subspace: remaining eigenvectors
    noise_subspace = eigenvectors[:, num_sources_subspace:]

    # Azimuth scan range (0 to 360 degrees)
    azimuth_scan = np.arange(0, 360, azimuth_resolution)
    music_spectrum = np.zeros(len(azimuth_scan))

    # MUSIC spatial spectrum
    for idx, azimuth in enumerate(azimuth_scan):
        # Compute steering vector for this direction
        a = compute_steering_vector(azimuth, mic_positions)

        # MUSIC pseudo-spectrum: P(θ) = 1 / (a^H * En * En^H * a)
        # where En is the noise subspace
        projection = np.dot(noise_subspace, np.conj(noise_subspace).T)
        denominator = np.dot(np.conj(a).T, np.dot(projection, a))

        # Avoid division by zero
        music_spectrum[idx] = 1.0 / (np.abs(denominator) + 1e-10)

    # Normalize spectrum
    music_spectrum = music_spectrum / (np.max(music_spectrum) + 1e-10)

    # Find peaks in MUSIC spectrum
    peaks, _ = find_peaks(
        music_spectrum,
        height=MIN_PEAK_HEIGHT,
        distance=int(min_separation / azimuth_resolution),
        prominence=MIN_PROMINENCE
    )

    # Sort peaks by power (strongest first)
    if len(peaks) > 0:
        peak_powers = music_spectrum[peaks]
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

        return sources, (azimuth_scan, music_spectrum), sound_detected
    else:
        return [], (azimuth_scan, music_spectrum), sound_detected


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

            # Detect multiple sources using MUSIC
            sources, spectrum_data, sound_detected = detect_multiple_sources_music(
                audio_data,
                num_sources=MAX_SOURCES,
                num_sources_subspace=NUM_SOURCES_MUSIC,
                min_separation=MIN_SEPARATION,
                azimuth_resolution=AZIMUTH_RESOLUTION
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
    ax1.set_title('Multi-Source DOA\n(MUSIC Algorithm)', fontsize=12, weight='bold')

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
    print("MULTI-SOURCE MUSIC ALGORITHM - 4x4 MICROPHONE ARRAY")
    print("=" * 80)
    print(f"Array configuration: 16 microphones in 4x4 grid")
    print(f"Microphone spacing: {MIC_SPACING*1000:.1f} mm")
    print(f"Origin: CENTER of array (0,0)")
    print(f"Detection method: MUSIC (Multiple Signal Classification)")
    print(f"Maximum sources detected: {MAX_SOURCES} (channels / 2)")
    print(f"Signal subspace dimension: {NUM_SOURCES_MUSIC}")
    print(f"Minimum separation: {MIN_SEPARATION}°")
    print(f"Azimuth resolution: {AZIMUTH_RESOLUTION}°")
    print(f"Peak detection threshold: {MIN_PEAK_HEIGHT}")
    print(f"Sound threshold: {SOUND_THRESHOLD}")
    print(f"Covariance smoothing: {SMOOTHING_FACTOR}")
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

    print(f"\nStarting real-time MUSIC-based multi-source detection...")
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
