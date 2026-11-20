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
ARRAY_DIAMETER = 0.090  # 90 mm diameter
ARRAY_RADIUS = ARRAY_DIAMETER / 2  # 45 mm radius

# Audio settings
SAMPLE_RATE = 48000  # Sample rate
CHANNELS = 8  # 8 microphones: 1 center + 7 on edge

# Multi-source detection settings
MAX_SOURCES = None  # Maximum number of sources (None = dynamic, detect all valid peaks)
MIN_SEPARATION = 15  # Minimum angular separation between sources (degrees)
AZIMUTH_RESOLUTION = 2  # Degrees (scanning resolution)
MIN_PEAK_HEIGHT = 0.3  # Minimum relative power for peak detection (0-1)
MIN_PROMINENCE = 0.15  # How much peak must stand out from background
SOUND_THRESHOLD = 0.002  # RMS threshold for sound detection (absolute RMS amplitude)

# Queue for audio data (non-blocking)
audio_queue = queue.Queue(maxsize=10)

# Audio buffer to accumulate samples
audio_buffer = []

# Latest detection results
latest_sources = []
latest_spectrum = None
sound_active = False


def get_mic_positions_circular():
    """
    Generate microphone positions for circular array with origin at CENTER.

    Array configuration:
    - 1 microphone at center (0, 0)
    - 7 microphones equally spaced on circle edge (45 mm radius)
    - Edge mics at angles: 0°, 51.43°, 102.86°, 154.29°, 205.71°, 257.14°, 308.57°
      (measured from +X axis, counter-clockwise)

    Channel mapping:
    - Channel 0: Center mic
    - Channels 1-7: Edge mics (starting from North, going clockwise)

    Returns:
        Dictionary mapping channel numbers to (x, y) positions in meters
    """
    mic_positions = {}

    # Channel 0: Center microphone
    mic_positions[0] = (0.0, 0.0)

    # Channels 1-7: Edge microphones
    # Starting from North (0°), going clockwise
    # In compass convention: 0° = North (+Y), 90° = East (+X)
    num_edge_mics = 7
    for i in range(num_edge_mics):
        # Angle in compass convention (0° = North, clockwise)
        compass_angle = i * (360.0 / num_edge_mics)

        # Convert to math convention (0° = East, counter-clockwise)
        math_angle_rad = np.radians(90 - compass_angle)

        x = ARRAY_RADIUS * np.cos(math_angle_rad)
        y = ARRAY_RADIUS * np.sin(math_angle_rad)

        mic_positions[i + 1] = (x, y)

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


def detect_multiple_sources_beamforming(audio_data, num_sources=MAX_SOURCES,
                                       min_separation=MIN_SEPARATION,
                                       azimuth_resolution=AZIMUTH_RESOLUTION):
    """
    Detect multiple sound sources using delay-and-sum beamforming and peak detection.

    This function scans all azimuth directions, applies beamforming to enhance signals
    from each direction, and detects multiple sources by finding peaks in the
    resulting power spectrum.

    Args:
        audio_data: Audio data array [samples, channels]
        num_sources: Maximum number of sources to detect (None = dynamic, detect all valid peaks)
        min_separation: Minimum angular separation between sources (degrees)
        azimuth_resolution: Angular resolution for scanning (degrees)

    Returns:
        sources: List of dictionaries with 'azimuth', 'power', 'confidence', 'id'
        spectrum_data: Tuple of (azimuth_scan, power_spectrum) for visualization
        sound_detected: Boolean indicating if sound was detected
    """
    # Normalize signals
    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)

    # Check if sound is detected
    avg_signal = np.mean(audio_data, axis=1)
    sound_detected = detect_sound(avg_signal, threshold=SOUND_THRESHOLD)

    if not sound_detected:
        return [], None, False

    # Get mic positions
    mic_positions = get_mic_positions_circular()

    # Azimuth scan range (0 to 360 degrees)
    azimuth_scan = np.arange(0, 360, azimuth_resolution)
    power_spectrum = np.zeros(len(azimuth_scan))

    num_samples = len(audio_data)

    # Beamforming: scan all directions
    for idx, azimuth in enumerate(azimuth_scan):
        # Convert azimuth to radians (accounting for coordinate system)
        # 0° = North, 90° = East, 180° = South, 270° = West
        theta_rad = np.radians(90 - azimuth)

        # Source direction unit vector
        source_direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

        # Calculate steering delays for each microphone
        delayed_signals = []
        for ch in range(CHANNELS):
            pos = np.array(mic_positions[ch])

            # Time delay for this mic relative to origin
            # Positive when source is in direction of microphone
            distance_proj = np.dot(pos, source_direction)
            time_delay = -distance_proj / SPEED_OF_SOUND

            # Apply delay (shift signal)
            delay_samples = int(time_delay * SAMPLE_RATE)

            # Shift signal by delay
            if delay_samples > 0:
                # Pad at beginning, remove from end
                delayed = np.pad(audio_data[:, ch], (delay_samples, 0), mode='constant')[:num_samples]
            elif delay_samples < 0:
                # Pad at end, remove from beginning
                delayed = np.pad(audio_data[:, ch], (0, -delay_samples), mode='constant')[-num_samples:]
            else:
                delayed = audio_data[:, ch]

            delayed_signals.append(delayed)

        # Sum aligned signals (delay-and-sum beamforming)
        beamformed = np.mean(delayed_signals, axis=0)

        # Calculate power in this direction
        power_spectrum[idx] = np.mean(beamformed**2)

    # Normalize power spectrum
    power_spectrum = power_spectrum / (np.max(power_spectrum) + 1e-10)

    # Find peaks in power spectrum
    # distance parameter ensures minimum separation between peaks
    peaks, _ = find_peaks(
        power_spectrum,
        height=MIN_PEAK_HEIGHT,  # Minimum relative power
        distance=int(min_separation / azimuth_resolution),  # Minimum samples between peaks
        prominence=MIN_PROMINENCE  # How much peak stands out from background
    )

    # Sort peaks by power (strongest first)
    if len(peaks) > 0:
        peak_powers = power_spectrum[peaks]
        sorted_indices = np.argsort(peak_powers)[::-1]
        peaks = peaks[sorted_indices]
        peak_powers = peak_powers[sorted_indices]

        # Limit to num_sources if specified (None = dynamic, keep all peaks)
        if num_sources is not None:
            peaks = peaks[:num_sources]
            peak_powers = peak_powers[:num_sources]

        # Create source list with azimuth from beamforming
        sources = []
        for peak_idx, power in zip(peaks, peak_powers):
            # Azimuth from beamforming peak
            azimuth = azimuth_scan[peak_idx]

            # Estimate confidence based on peak power
            confidence = float(power)

            sources.append({
                'azimuth': float(azimuth),
                'elevation': 0.0,  # Horizontal plane only
                'power': float(power),
                'confidence': confidence,
                'id': len(sources)  # Simple ID for coloring
            })

        return sources, (azimuth_scan, power_spectrum), sound_detected
    else:
        return [], (azimuth_scan, power_spectrum), sound_detected


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
    buffer_duration = 0.5  # seconds
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
        if len(audio_data) >= int(buffer_duration * SAMPLE_RATE):
            # Take last buffer_duration worth of samples
            audio_data = audio_data[-int(buffer_duration * SAMPLE_RATE):]

            # Remove DC offset from each channel
            audio_data = audio_data - np.mean(audio_data, axis=0)

            # Detect multiple sources using beamforming
            sources, spectrum_data, sound_detected = detect_multiple_sources_beamforming(
                audio_data,
                num_sources=MAX_SOURCES,
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
    ax1.set_title('Multi-Source DOA\n(Circular Array Beamforming)', fontsize=12, weight='bold')

    # Draw microphone positions on polar plot
    mic_positions = get_mic_positions_circular()
    for ch, pos in mic_positions.items():
        if ch == 0:
            # Center mic
            ax1.plot(0, 0, 'ko', markersize=8, label='Center mic')
        else:
            # Edge mics - convert (x,y) to polar
            r = np.sqrt(pos[0]**2 + pos[1]**2)
            theta = np.arctan2(pos[0], pos[1])  # Note: atan2(x,y) for compass convention
            # Scale radius for display
            ax1.plot(theta, 0.15, 'bs', markersize=6, alpha=0.5)

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
        ax1.text(0, 0.5, 'No sound\ndetected',
                ha='center', va='center', fontsize=12, color='gray')
        ax1.set_ylim(0, 1)

    # ===== Plot 2: Array visualization with DOA =====
    # Plot microphone array layout
    mic_positions = get_mic_positions_circular()

    # Draw array circle
    circle_theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(ARRAY_RADIUS * np.cos(circle_theta) * 1000,
             ARRAY_RADIUS * np.sin(circle_theta) * 1000,
             'k--', alpha=0.3, linewidth=1)

    # Plot microphones
    for ch, pos in mic_positions.items():
        if ch == 0:
            ax2.scatter(pos[0] * 1000, pos[1] * 1000, c='black', s=100, marker='o',
                       label='Center mic', zorder=5)
        else:
            ax2.scatter(pos[0] * 1000, pos[1] * 1000, c='blue', s=80, marker='s',
                       alpha=0.7, zorder=5)
            ax2.annotate(f'{ch}', (pos[0] * 1000, pos[1] * 1000),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    if sound_active and len(latest_sources) > 0:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow', 'magenta']

        for idx, source in enumerate(latest_sources):
            azimuth_rad = np.radians(source['azimuth'])
            color = colors[idx % len(colors)]

            # Direction vector (0° = North = +Y, 90° = East = +X)
            dx = np.sin(azimuth_rad) * 60  # Scale for display (mm)
            dy = np.cos(azimuth_rad) * 60

            ax2.arrow(0, 0, dx, dy, head_width=8, head_length=5,
                     fc=color, ec=color, linewidth=2, alpha=0.8)
            ax2.text(dx * 1.2, dy * 1.2, f"S{idx+1}: {source['azimuth']:.0f}°",
                    fontsize=9, color=color, weight='bold', ha='center')

    ax2.set_xlabel('X (East) [mm]', fontsize=10)
    ax2.set_ylabel('Y (North) [mm]', fontsize=10)
    ax2.set_title(f'Circular Array Layout (Ø{ARRAY_DIAMETER*1000:.0f}mm)\n1 center + 7 edge mics',
                 fontsize=11, weight='bold')
    ax2.set_xlim(-80, 80)
    ax2.set_ylim(-80, 80)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)


if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-SOURCE BEAMFORMING - CIRCULAR MICROPHONE ARRAY")
    print("=" * 80)
    print(f"Array configuration: {CHANNELS} microphones (1 center + 7 edge)")
    print(f"Array diameter: {ARRAY_DIAMETER*1000:.0f} mm (radius: {ARRAY_RADIUS*1000:.0f} mm)")
    print(f"Edge mic spacing: {360/7:.2f}° apart")
    print(f"Detection method: Delay-and-Sum Beamforming")
    print(f"Maximum sources: {'Dynamic (all valid peaks)' if MAX_SOURCES is None else MAX_SOURCES}")
    print(f"Minimum separation: {MIN_SEPARATION}°")
    print(f"Azimuth resolution: {AZIMUTH_RESOLUTION}°")
    print(f"Peak detection threshold: {MIN_PEAK_HEIGHT}")
    print(f"Sound threshold: {SOUND_THRESHOLD} (RMS amplitude)")
    print("Note: All sources detected in horizontal plane (elevation = 0°)")
    print("=" * 80)

    # Print mic positions
    print("\nMicrophone positions:")
    mic_positions = get_mic_positions_circular()
    for ch, pos in mic_positions.items():
        if ch == 0:
            print(f"  Ch {ch}: CENTER (0.0, 0.0) mm")
        else:
            angle = (ch - 1) * (360 / 7)
            print(f"  Ch {ch}: ({pos[0]*1000:6.2f}, {pos[1]*1000:6.2f}) mm  [angle: {angle:.1f}°]")
    print()

    # Find audio device
    devices = sd.query_devices()
    audio_device = None

    for i, device in enumerate(devices):
        if 'line' in device['name'].lower():
            audio_device = i
            print(f"Found audio device at index {i}")
            print(f"  Name: {device['name']}")
            print(f"  Max input channels: {device['max_input_channels']}")
            print(f"  Default sample rate: {device['default_samplerate']}")
            break

    if audio_device is None:
        print("Warning: No 'line' device found. Using default input device.")
        audio_device = sd.default.device[0]
        device_info = sd.query_devices(audio_device)
        print(f"  Using: {device_info['name']}")
        print(f"  Max input channels: {device_info['max_input_channels']}")

    print(f"\nStarting real-time multi-source beamforming...")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS}")

    # Start audio stream
    stream = sd.InputStream(
        device=audio_device,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1)
    )

    stream.start()

    # Set up matplotlib figure
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='polar')
    ax2 = fig.add_subplot(122)

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
