import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.fft import fft, ifft, fftshift
import queue

# Constants for DOA calculation
SPEED_OF_SOUND = 343.0  # m/s at room temperature
MIC_ARRAY_RADIUS = 0.045  # meters (adjust to your actual array radius)
MIC_SPACING = 0.045  # meters (spacing between mics in rectangular/line arrays)

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 8
BUFFER_DURATION = 0.5  # seconds of audio to process
SOUND_THRESHOLD = 0.08  # RMS threshold for sound detection

# Channel configuration
PDM_CHANNEL = 0  # PDM mic for sound detection
MEMS_CHANNELS = list(range(1, 8))  # MEMS mics for DOA

# Queue for audio data (non-blocking)
audio_queue = queue.Queue(maxsize=10)

# Audio buffer to accumulate samples
audio_buffer = []

# Latest DOA results
latest_azimuth = None
latest_time_delays = None
sound_active = False

# Array geometry selection
ARRAY_TYPE = 'circular'  # Will be set by user selection


def get_mic_positions(array_type, num_mics=7):
    """
    Generate microphone positions for different array geometries.

    Args:
        array_type: Type of array ('circular', 'rectangular_4x2', 'line_horizontal',
                   'line_vertical', 'plus')
        num_mics: Number of microphones (excluding reference mic at origin)

    Returns:
        List of (x, y) tuples representing mic positions in meters
        Reference mic is always at (0, 0) and not included in the list
    """
    mic_positions = []

    if array_type == 'circular':
        # Circular array: mics distributed evenly around a circle
        for i in range(num_mics):
            angle = 2 * np.pi * i / num_mics
            x = MIC_ARRAY_RADIUS * np.cos(angle)
            y = MIC_ARRAY_RADIUS * np.sin(angle)
            mic_positions.append((x, y))

    elif array_type == 'rectangular_4x2':
        # 4x2 rectangular array (8 mics total including reference)
        # Reference at (0, 0), then 7 mics in a 4x2 grid pattern
        # Layout:  x x x x
        #          o x x x  (o is reference at origin)
        positions = [
            (MIC_SPACING, 0),
            (2 * MIC_SPACING, 0),
            (3 * MIC_SPACING, 0),
            (0, MIC_SPACING),
            (MIC_SPACING, MIC_SPACING),
            (2 * MIC_SPACING, MIC_SPACING),
            (3 * MIC_SPACING, MIC_SPACING)
        ]
        mic_positions = positions[:num_mics]

    elif array_type == 'line_horizontal':
        # 8x1 horizontal line array
        # Layout: o x x x x x x x
        for i in range(num_mics):
            x = (i + 1) * MIC_SPACING
            y = 0
            mic_positions.append((x, y))

    elif array_type == 'line_vertical':
        # 1x8 vertical line array
        # Layout: x
        #         x
        #         x
        #         x
        #         x
        #         x
        #         x
        #         o
        for i in range(num_mics):
            x = 0
            y = (i + 1) * MIC_SPACING
            mic_positions.append((x, y))

    elif array_type == 'plus':
        # '+' shaped array: 4x1 horizontal and 1x4 vertical (8 mics total)
        # Layout:     x
        #             x
        #             x
        #         x x o x
        #             x
        #             x
        #             x
        positions = [
            (MIC_SPACING, 0),      # Right
            (-MIC_SPACING, 0),     # Left
            (-2 * MIC_SPACING, 0), # Far left
            (0, MIC_SPACING),      # Up
            (0, 2 * MIC_SPACING),  # Far up
            (0, 3 * MIC_SPACING),  # Farthest up
            (0, -MIC_SPACING)      # Down
        ]
        mic_positions = positions[:num_mics]

    return mic_positions


def detect_sound(pdm_signal, threshold=SOUND_THRESHOLD):
    """
    Detect if there's a sound event in the PDM signal.

    Args:
        pdm_signal: Audio signal from PDM mic
        threshold: RMS threshold for sound detection

    Returns:
        bool: True if sound detected, False otherwise
    """
    rms = np.sqrt(np.mean(pdm_signal**2))
    return rms > threshold


def calculate_doa(time_delays, mic_positions):
    """
    Calculate Direction of Arrival using time delays.

    Args:
        time_delays: Array of time delays (in seconds) relative to reference mic
        mic_positions: Array of mic positions in 2D [(x, y), ...] in meters
                      relative to reference mic at (0, 0)

    Returns:
        angle: Azimuth angle in degrees (0° is North, clockwise)
    """
    # Use weighted average of angle estimates from each microphone
    angles = []
    weights = []

    for i, (td, pos) in enumerate(zip(time_delays, mic_positions)):
        # Calculate distance difference
        distance_diff = td * SPEED_OF_SOUND

        # Calculate actual distance from reference mic (at origin) to this mic
        mic_distance = np.sqrt(pos[0]**2 + pos[1]**2)

        # Get mic angle from reference position
        mic_angle = np.arctan2(pos[1], pos[0])

        # The time delay tells us if sound arrives early (positive) or late (negative)
        # at this mic compared to reference
        # Maximum possible delay is when sound travels along the mic baseline
        if abs(distance_diff) < mic_distance:
            # Use cosine rule: d = baseline * cos(theta)
            # where theta is angle between source direction and mic-to-mic baseline
            cos_theta = distance_diff / mic_distance
            cos_theta = np.clip(cos_theta, -1, 1)  # Ensure valid range

            # Source angle calculation:
            # Positive delay means sound arrives at this mic later (source closer to reference)
            # This means source is on the reference side of the baseline
            source_angle = mic_angle - np.arccos(cos_theta)

            angles.append(source_angle)
            weights.append(abs(td))  # Weight by time delay magnitude

    if len(angles) > 0 and sum(weights) > 0:
        # Circular mean for angles
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


def process_audio_buffer(audio_data):
    """
    Process audio buffer to extract DOA using frequency domain correlation.

    Args:
        audio_data: Audio data array [samples, channels]

    Returns:
        azimuth: Direction of arrival in degrees (None if no sound detected)
        time_delays: Array of time delays for each MEMS mic
        sound_detected: Boolean indicating if sound was detected
    """
    # Normalize signals
    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)

    pdm_signal = audio_data[:, PDM_CHANNEL]
    N = len(pdm_signal)

    # Check if sound is detected using PDM mic
    sound_detected = detect_sound(pdm_signal, threshold=SOUND_THRESHOLD)

    if not sound_detected:
        return None, None, False

    # Use first MEMS channel as reference for cross-correlation
    ref_channel = MEMS_CHANNELS[0]
    ref_signal = audio_data[:, ref_channel]

    time_delays = []

    # Get mic positions based on selected array type
    # Reference mic (MEMS_CHANNELS[0]) is at position (0, 0)
    # Remaining mics are positioned according to array geometry
    mic_positions = get_mic_positions(ARRAY_TYPE, num_mics=len(MEMS_CHANNELS)-1)

    # Cross-correlate each MEMS mic (except reference) with the reference MEMS mic
    # Using frequency domain method (GCC-PHAT)
    N_fft = 2 * N - 1

    # FFT of reference signal
    freq_ref = fft(ref_signal, N_fft)

    for mems_ch in MEMS_CHANNELS[1:]:
        mems_signal = audio_data[:, mems_ch]

        # Frequency domain cross-correlation
        freq_mems = fft(mems_signal, N_fft)

        # Cross-power spectrum with normalization (GCC-PHAT)
        cross_spectrum = np.conjugate(freq_ref) * freq_mems
        cross_spectrum_norm = cross_spectrum / (np.abs(cross_spectrum) + 1e-10)

        # Inverse FFT to get correlation
        correlation = ifft(cross_spectrum_norm)
        correlation = fftshift(correlation)

        lags = np.arange(-N_fft//2, N_fft//2)

        # Find peak
        peak_idx = np.argmax(correlation.real)
        peak_lag = lags[peak_idx]
        peak_time = peak_lag / SAMPLE_RATE  # seconds

        time_delays.append(peak_time)

    # Calculate DOA
    azimuth = calculate_doa(np.array(time_delays), mic_positions)

    return azimuth, time_delays, sound_detected


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
            print(f"Buffer size: {len(audio_data)} samples, RMS: {np.sqrt(np.mean(audio_data[:, PDM_CHANNEL]**2)):.4f}")

        # Only process if we have enough samples
        if len(audio_data) >= int(BUFFER_DURATION * SAMPLE_RATE):
            # Take last BUFFER_DURATION worth of samples
            audio_data = audio_data[-int(BUFFER_DURATION * SAMPLE_RATE):]

            azimuth, time_delays, sound_detected = process_audio_buffer(audio_data)

            if sound_detected and azimuth is not None:
                latest_azimuth = azimuth
                latest_time_delays = time_delays
                sound_active = True
                print(f">>> DOA: {azimuth:.1f}° (Frequency Domain)")
            else:
                sound_active = False

    # Clear previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Polar plot for DOA
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_title('Direction of Arrival (Real-time, Freq Domain)')

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

    # Time delays plot
    if sound_active and latest_time_delays is not None:
        mems_comparison_channels = MEMS_CHANNELS[1:]
        ax2.bar(range(len(latest_time_delays)), np.array(latest_time_delays) * 1e6, color='blue')
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax2.set_xticks(range(len(latest_time_delays)))
        ax2.set_xticklabels([f'M{ch}' for ch in mems_comparison_channels])
        ax2.set_xlabel(f'MEMS Channel (vs M{MEMS_CHANNELS[0]} reference)')
        ax2.set_ylabel('Time Delay (μs)')
        ax2.set_title('Time Delays Between MEMS Mics (Freq Domain)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-200, 200)  # Adjust based on your expected delays
    else:
        ax2.text(0.5, 0.5, 'Waiting for sound...',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, color='gray')
        ax2.set_xlim(0, 6)
        ax2.set_ylim(-200, 200)

    # Power plot
    if sound_active and latest_time_delays is not None:
        ax3.plot(np.abs(audio_data), linestyle=(0, (1, 10)))
        ax3.plot(np.ones(np.size(audio_data, 0))*SOUND_THRESHOLD, color='red', linestyle=':')
        ax3.set_xlabel('Sample')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Amplitude vs Time')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.003, 0.003)  # Adjust based on your expected delays
    else:
        ax3.text(0.5, 0.5, 'Waiting for sound...',
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=12, color='gray')
        ax3.set_xlim(0, 6)
        ax3.set_ylim(-0.003, 0.003)


if __name__ == "__main__":
    # Array selection menu
    print("=" * 60)
    print("SELECT MICROPHONE ARRAY GEOMETRY")
    print("=" * 60)
    print("1. Circular array (default)")
    print("2. Rectangular 4x2 array")
    print("3. Horizontal line array (8x1)")
    print("4. Vertical line array (1x8)")
    print("5. Plus (+) array (4x1 by 1x4)")
    print("=" * 60)

    array_options = {
        '1': 'circular',
        '2': 'rectangular_4x2',
        '3': 'line_horizontal',
        '4': 'line_vertical',
        '5': 'plus'
    }

    while True:
        choice = input("Enter your choice (1-5): ").strip()
        if choice in array_options:
            ARRAY_TYPE = array_options[choice]
            print(f"\nSelected: {ARRAY_TYPE}")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

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

    print(f"\nStarting real-time DOA estimation (Frequency Domain)...")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS}")
    print(f"PDM channel (trigger): {PDM_CHANNEL}")
    print(f"MEMS channels (DOA): {MEMS_CHANNELS}")
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
    fig = plt.figure(figsize=(14, 6))
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
