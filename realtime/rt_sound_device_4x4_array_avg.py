import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.signal import correlate
import queue

# Constants for DOA calculation
SPEED_OF_SOUND = 343.0  # m/s at room temperature
MIC_SPACING = 0.042  # meters (spacing between mics in the 4x4 array)

# Audio settings
SAMPLE_RATE = 48000  # Sample rate for 16-channel array
CHANNELS = 16  # 16 microphones in 4x4 array
BUFFER_DURATION = 0.5  # seconds of audio to process
SOUND_THRESHOLD = 0.002  # RMS threshold for sound detection

# Channel configuration for 4x4 array
# MIC01 = channel 0, MIC02 = channel 1, etc.
# Row-based adaptive references: MIC15(14), MIC14(13), MIC06(5), MIC07(6)
ALL_CHANNELS = list(range(16))  # All 16 microphones (channels 0-15 = MIC01-MIC16)

# Queue for audio data (non-blocking)
audio_queue = queue.Queue(maxsize=10)

# Audio buffer to accumulate samples
audio_buffer = []

# Latest DOA results
latest_azimuth = None
latest_time_delays = None
sound_active = False


def get_mic_positions_4x4():
    """
    Generate microphone positions for 4x4 array with origin at CENTER of array.

    Physical layout - actual configuration:
    - Outer perimeter (12 mics), clockwise from bottom-left:
      MIC02, MIC04, MIC06, MIC08, MIC07, MIC10, MIC09, MIC11, MIC13, MIC15, MIC16, MIC01
    - Inner square (4 mics), clockwise from bottom-left:
      MIC03, MIC05, MIC12, MIC14

    Channel mapping: channel 0 = MIC01, channel 1 = MIC02, etc.

    Returns:
        Dictionary mapping channel numbers to (x, y) positions in meters
        Origin (0, 0) is at the CENTER of the array (between rows 1 & 2, columns 1 & 2)
    """
    # Spacing between outer perimeter mics
    outer_spacing = MIC_SPACING

    # Center of array is at 1.5 * outer_spacing from bottom-left corner in original coords
    # We'll shift everything so origin is at the center
    center_offset_x = 1.5 * outer_spacing
    center_offset_y = 1.5 * outer_spacing

    # Inner square spacing
    inner_half_spacing = outer_spacing / 2

    # Define all positions with origin at CENTER of array
    # Each position is shifted by -center_offset to put origin at center

    mic_positions = {
        # Outer perimeter - Bottom edge
        1: (0 - center_offset_x, 0 - center_offset_y),                            # MIC02 (channel 1) - bottom-left corner
        3: (outer_spacing - center_offset_x, 0 - center_offset_y),                # MIC04 (channel 3)
        5: (2 * outer_spacing - center_offset_x, 0 - center_offset_y),            # MIC06 (channel 5)
        7: (3 * outer_spacing - center_offset_x, 0 - center_offset_y),            # MIC08 (channel 7) - bottom-right corner

        # Outer perimeter - Right edge
        6: (3 * outer_spacing - center_offset_x, outer_spacing - center_offset_y),        # MIC07 (channel 6)
        9: (3 * outer_spacing - center_offset_x, 2 * outer_spacing - center_offset_y),    # MIC10 (channel 9)

        # Outer perimeter - Top edge
        8: (3 * outer_spacing - center_offset_x, 3 * outer_spacing - center_offset_y),    # MIC09 (channel 8) - top-right corner
        10: (2 * outer_spacing - center_offset_x, 3 * outer_spacing - center_offset_y),   # MIC11 (channel 10)
        12: (outer_spacing - center_offset_x, 3 * outer_spacing - center_offset_y),       # MIC13 (channel 12)
        14: (0 - center_offset_x, 3 * outer_spacing - center_offset_y),                   # MIC15 (channel 14) - top-left corner

        # Outer perimeter - Left edge
        15: (0 - center_offset_x, 2 * outer_spacing - center_offset_y),                   # MIC16 (channel 15)
        0: (0 - center_offset_x, outer_spacing - center_offset_y),                        # MIC01 (channel 0)

        # Inner square (centered around origin)
        # MIC03, MIC05, MIC12, MIC14
        2: (-inner_half_spacing, -inner_half_spacing),      # MIC03 (channel 2) - inner bottom-left
        4: (inner_half_spacing, -inner_half_spacing),       # MIC05 (channel 4) - inner bottom-right
        11: (inner_half_spacing, inner_half_spacing),       # MIC12 (channel 11) - inner top-right
        13: (-inner_half_spacing, inner_half_spacing)       # MIC14 (channel 13) - inner top-left
    }

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


def calculate_doa(time_delays, mic_positions):
    """
    Calculate Direction of Arrival using time delays.

    Args:
        time_delays: Dictionary mapping channel numbers to time delays (in seconds)
                    relative to reference mic
        mic_positions: Dictionary mapping channel numbers to (x, y) positions in meters

    Returns:
        angle: Azimuth angle in degrees (0° is North, clockwise)
    """
    angles = []
    weights = []

    for ch, td in time_delays.items():
        pos = mic_positions[ch]

        # Calculate distance difference
        distance_diff = td * SPEED_OF_SOUND

        # Calculate actual distance from reference mic (at origin) to this mic
        mic_distance = np.sqrt(pos[0]**2 + pos[1]**2)

        # Avoid division by zero for reference mic
        if mic_distance < 1e-6:
            continue

        # Get mic angle from reference position
        mic_angle = np.arctan2(pos[1], pos[0])

        # The time delay tells us if sound arrives early (positive) or late (negative)
        # at this mic compared to reference
        if abs(distance_diff) < mic_distance:
            # Use cosine rule: d = baseline * cos(theta)
            cos_theta = distance_diff / mic_distance
            cos_theta = np.clip(cos_theta, -1, 1)  # Ensure valid range

            # Source angle calculation
            source_angle = mic_angle - np.arccos(cos_theta)

            angles.append(source_angle)
            weights.append(abs(td))  # Weight by time delay magnitude

    if len(angles) > 0 and sum(weights) > 0:
        # Circular mean for angles (weighted average)
        angles = np.array(angles)
        weights = np.array(weights)

        # Convert to unit vectors, average, then back to angle
        x_avg = np.average(np.cos(angles), weights=weights)
        y_avg = np.average(np.sin(angles), weights=weights)

        avg_angle = np.arctan2(y_avg, x_avg)

        # Convert to degrees (0° = East, counter-clockwise)
        # Then convert to compass bearing (0° = North, clockwise)
        azimuth = (90 - np.degrees(avg_angle)) % 360

        return azimuth

    return None


def get_4x4_matrix_layout():
    """
    Define the 4x4 matrix layout for systematic row/column cross-correlation.

    Matrix layout (rows from bottom to top):
    Row 1 (bottom): [1, 0, 15, 14]  - MIC02, MIC01, MIC16, MIC15
    Row 2:          [2, 3, 13, 12]  - MIC03, MIC04, MIC14, MIC13
    Row 3:          [4, 5, 11, 10]  - MIC05, MIC06, MIC12, MIC11
    Row 4 (top):    [6, 7, 9, 8]    - MIC07, MIC08, MIC10, MIC09

    Returns:
        2D numpy array (4x4) with channel numbers
    """
    matrix = np.array([
        [6, 7, 9, 8],      # Row 3 (top): channels for MIC07, MIC08, MIC10, MIC09
        [4, 5, 11, 10],    # Row 2: channels for MIC05, MIC06, MIC12, MIC11
        [2, 3, 13, 12],    # Row 1: channels for MIC03, MIC04, MIC14, MIC13
        [1, 0, 15, 14]     # Row 0 (bottom): channels for MIC02, MIC01, MIC16, MIC15
    ])

    return matrix


def process_audio_buffer(audio_data):
    """
    Process audio buffer to extract DOA for 4x4 array using adaptive reference.

    Each row calculates DOA independently with its reference mic, then
    the final DOA is computed using SIMPLE CIRCULAR AVERAGE (like original script).

    Reference microphones for rows: MIC15(14), MIC14(13), MIC06(5), MIC07(6)

    Args:
        audio_data: Audio data array [samples, channels]

    Returns:
        azimuth: Direction of arrival in degrees (circular average of row DOAs)
        row_azimuths: Dictionary of DOA values for each row
        sound_detected: Boolean indicating if sound was detected
    """
    # Normalize signals
    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)

    # Use average of all channels for sound detection (more robust)
    avg_signal = np.mean(audio_data, axis=1)

    # Check if sound is detected
    sound_detected = detect_sound(avg_signal, threshold=SOUND_THRESHOLD)

    if not sound_detected:
        return None, None, False

    # Get mic positions
    mic_positions = get_mic_positions_4x4()

    # Get 4x4 matrix layout
    matrix = get_4x4_matrix_layout()

    # Store DOA values for each row
    row_azimuths = {}
    all_time_delays = {}  # For visualization

    # === ROW-WISE PROCESSING ===
    # Process each row independently

    # Row 0 (bottom): [1, 0, 15, 14] - use channel 14 (MIC15) as reference
    row_0_ref = 14
    row_0_channels = [1, 0, 15]
    row_0_signal = audio_data[:, row_0_ref]
    row_0_delays = {row_0_ref: 0.0}

    for ch in row_0_channels:
        mic_signal = audio_data[:, ch]
        correlation = correlate(mic_signal, row_0_signal, mode='same')
        lags = np.arange(-len(row_0_signal)//2, len(row_0_signal)//2)
        peak_idx = np.argmax(np.abs(correlation))
        peak_lag = lags[peak_idx]
        row_0_delays[ch] = peak_lag / SAMPLE_RATE

    # Get positions for this row relative to row reference
    row_0_positions = {ch: mic_positions[ch] for ch in [row_0_ref] + row_0_channels}
    row_0_azimuth = calculate_doa(row_0_delays, row_0_positions)
    if row_0_azimuth is not None:
        row_azimuths['row_0'] = row_0_azimuth
    all_time_delays.update(row_0_delays)

    # Row 1: [2, 3, 13, 12] - use channel 13 (MIC14) as reference
    row_1_ref = 13
    row_1_channels = [2, 3, 12]
    row_1_signal = audio_data[:, row_1_ref]
    row_1_delays = {row_1_ref: 0.0}

    for ch in row_1_channels:
        mic_signal = audio_data[:, ch]
        correlation = correlate(mic_signal, row_1_signal, mode='same')
        lags = np.arange(-len(row_1_signal)//2, len(row_1_signal)//2)
        peak_idx = np.argmax(np.abs(correlation))
        peak_lag = lags[peak_idx]
        row_1_delays[ch] = peak_lag / SAMPLE_RATE

    row_1_positions = {ch: mic_positions[ch] for ch in [row_1_ref] + row_1_channels}
    row_1_azimuth = calculate_doa(row_1_delays, row_1_positions)
    if row_1_azimuth is not None:
        row_azimuths['row_1'] = row_1_azimuth
    all_time_delays.update(row_1_delays)

    # Row 2: [4, 5, 11, 10] - use channel 5 (MIC06) as reference
    row_2_ref = 5
    row_2_channels = [4, 11, 10]
    row_2_signal = audio_data[:, row_2_ref]
    row_2_delays = {row_2_ref: 0.0}

    for ch in row_2_channels:
        mic_signal = audio_data[:, ch]
        correlation = correlate(mic_signal, row_2_signal, mode='same')
        lags = np.arange(-len(row_2_signal)//2, len(row_2_signal)//2)
        peak_idx = np.argmax(np.abs(correlation))
        peak_lag = lags[peak_idx]
        row_2_delays[ch] = peak_lag / SAMPLE_RATE

    row_2_positions = {ch: mic_positions[ch] for ch in [row_2_ref] + row_2_channels}
    row_2_azimuth = calculate_doa(row_2_delays, row_2_positions)
    if row_2_azimuth is not None:
        row_azimuths['row_2'] = row_2_azimuth
    all_time_delays.update(row_2_delays)

    # Row 3 (top): [6, 7, 9, 8] - use channel 6 (MIC07) as reference
    row_3_ref = 6
    row_3_channels = [7, 9, 8]
    row_3_signal = audio_data[:, row_3_ref]
    row_3_delays = {row_3_ref: 0.0}

    for ch in row_3_channels:
        mic_signal = audio_data[:, ch]
        correlation = correlate(mic_signal, row_3_signal, mode='same')
        lags = np.arange(-len(row_3_signal)//2, len(row_3_signal)//2)
        peak_idx = np.argmax(np.abs(correlation))
        peak_lag = lags[peak_idx]
        row_3_delays[ch] = peak_lag / SAMPLE_RATE

    row_3_positions = {ch: mic_positions[ch] for ch in [row_3_ref] + row_3_channels}
    row_3_azimuth = calculate_doa(row_3_delays, row_3_positions)
    if row_3_azimuth is not None:
        row_azimuths['row_3'] = row_3_azimuth
    all_time_delays.update(row_3_delays)

    # === SIMPLE CIRCULAR AVERAGE OF ROW DOA VALUES (like original script) ===
    if len(row_azimuths) > 0:
        # Convert angles to radians
        row_angles_rad = [np.radians(az) for az in row_azimuths.values()]

        # Simple circular mean (NOT RMS)
        # Convert to unit vectors and average
        x_components = [np.cos(angle) for angle in row_angles_rad]
        y_components = [np.sin(angle) for angle in row_angles_rad]

        # Simple average (not RMS)
        x_avg = np.mean(x_components)
        y_avg = np.mean(y_components)

        # Convert back to angle
        combined_angle_rad = np.arctan2(y_avg, x_avg)
        combined_azimuth = (90 - np.degrees(combined_angle_rad)) % 360

        return combined_azimuth, row_azimuths, sound_detected
    else:
        return None, row_azimuths, sound_detected


def audio_callback(indata, frames, time, status):
    """
    Callback function for sounddevice stream.
    Called for each audio block received from the device.
    """
    if status:
        print(f"Status: {status}")

    # Put data in queue (non-blocking, discard if full)
    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        pass  # Discard if queue is full


def update_plot(frame):
    """
    Update function for matplotlib animation.
    """
    global latest_azimuth, latest_time_delays, sound_active, audio_buffer

    # Get all available audio chunks from queue
    try:
        while True:
            chunk = audio_queue.get_nowait()
            audio_buffer.append(chunk)
    except queue.Empty:
        pass

    # Keep buffer size manageable (last 1 second of audio)
    max_buffer_samples = int(SAMPLE_RATE * 1.0)
    if len(audio_buffer) > 0:
        total_samples = sum(len(chunk) for chunk in audio_buffer)

        # Remove old chunks if buffer is too large
        while total_samples > max_buffer_samples and len(audio_buffer) > 1:
            removed = audio_buffer.pop(0)
            total_samples -= len(removed)

    # Process if we have enough samples
    if len(audio_buffer) > 0:
        audio_data = np.vstack(audio_buffer)

        # Debug: print buffer size
        if frame % 10 == 0:  # Print every 10 frames
            # Calculate average RMS across all channels
            avg_rms = np.sqrt(np.mean(np.mean(audio_data**2, axis=1)))
            print(f"Buffer size: {len(audio_data)} samples, Avg RMS: {avg_rms:.4f}")

        # Only process if we have enough samples
        if len(audio_data) >= int(BUFFER_DURATION * SAMPLE_RATE):
            # Take last BUFFER_DURATION worth of samples
            audio_data = audio_data[-int(BUFFER_DURATION * SAMPLE_RATE):]

            azimuth, row_azimuths, sound_detected = process_audio_buffer(audio_data)

            if sound_detected and azimuth is not None:
                latest_azimuth = azimuth
                latest_time_delays = row_azimuths  # Store row DOA values
                sound_active = True
                # Print combined DOA and individual row DOAs
                row_info = ", ".join([f"{k}: {v:.1f}°" for k, v in row_azimuths.items()])
                print(f">>> Combined DOA: {azimuth:.1f}° | Row DOAs: {row_info}")
            else:
                sound_active = False

    # Clear previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Polar plot for DOA
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_title('Direction of Arrival - 4x4 Array [AVERAGING] (Real-time)')

    if sound_active and latest_azimuth is not None:
        # Plot DOA as arrow
        azimuth_rad = np.radians(latest_azimuth)
        ax1.arrow(azimuth_rad, 0, 0, 0.8,
                 head_width=0.3, head_length=0.1,
                 fc='red', ec='red', linewidth=2)
        ax1.text(azimuth_rad, 0.95, f'{latest_azimuth:.1f}°',
                ha='center', va='bottom', fontsize=12, color='red', weight='bold')
    else:
        ax1.text(0, 0, 'No sound\ndetected',
                ha='center', va='center', fontsize=12, color='gray')

    ax1.set_ylim(0, 1)

    # Row DOA plot
    if sound_active and latest_time_delays is not None:
        # latest_time_delays now contains row_azimuths dictionary
        row_names = sorted(latest_time_delays.keys())
        row_doas = [latest_time_delays[row] for row in row_names]

        ax2.bar(range(len(row_doas)), row_doas, color=['blue', 'green', 'orange', 'purple'][:len(row_doas)])
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax2.set_xticks(range(len(row_names)))
        ax2.set_xticklabels([name.replace('_', ' ').title() for name in row_names], rotation=45)
        ax2.set_xlabel('Row')
        ax2.set_ylabel('DOA (degrees)')
        ax2.set_title('Direction of Arrival per Row')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 360)  # Full circle
    else:
        ax2.text(0.5, 0.5, 'Waiting for sound...',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, color='gray')
        ax2.set_xlim(0, 4)
        ax2.set_ylim(0, 360)

    # Amplitude plot
    if sound_active and latest_time_delays is not None:
        ax3.plot(np.abs(audio_data), linestyle=(0, (1, 10)), alpha=0.5)
        ax3.axhline(y=SOUND_THRESHOLD, color='red', linestyle=':', label=f'Threshold: {SOUND_THRESHOLD}')
        ax3.set_xlabel('Sample')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Signal Amplitude vs Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(0, 0.5)
    else:
        ax3.text(0.5, 0.5, 'Waiting for sound...',
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=12, color='gray')
        ax3.set_xlim(0, 1000)
        ax3.set_ylim(0, 0.5)


if __name__ == "__main__":
    print("=" * 60)
    print("4x4 MICROPHONE ARRAY - DOA (SIMPLE AVERAGING)")
    print("=" * 60)
    print(f"Array configuration: 16 microphones")
    print(f"Origin: CENTER of array (0,0)")
    print(f"Processing: Row-based adaptive reference with CIRCULAR AVERAGING")
    print(f"Microphone spacing: {MIC_SPACING*1000:.1f} mm")
    print("=" * 60)
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

    print(f"\nStarting real-time DOA estimation...")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS}")
    print(f"Row references: MIC15(14), MIC14(13), MIC06(5), MIC07(6)")
    print(f"Sound threshold: {SOUND_THRESHOLD}")
    print(f"Buffer duration: {BUFFER_DURATION} seconds")

    # Start audio stream
    stream = sd.InputStream(
        device=minidsp_device,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1)  # 100ms blocks
    )

    stream.start()

    # Set up matplotlib figure
    fig = plt.figure(figsize=(16, 6))
    ax1 = plt.subplot(2, 2, 1, projection='polar')
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)

    # Create animation with longer interval to reduce processing load
    ani = animation.FuncAnimation(
        fig, update_plot, interval=200, cache_frame_data=False, blit=False
    )

    plt.tight_layout()

    try:
        print("\nReal-time processing active. Close the plot window to stop.")
        plt.show()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop()
        stream.close()
        print("Stream closed.")
