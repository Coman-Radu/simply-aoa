import sounddevice as sd
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better Windows compatibility
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import queue

# Audio settings
SAMPLE_RATE = 48000  # Sample rate for 16-channel array
CHANNELS = 16  # 16 microphones in 4x4 array
DISPLAY_DURATION = 0.05  # seconds of audio to display (50ms)
BLOCKSIZE = int(SAMPLE_RATE * 0.1)  # 100ms blocks
SOUND_THRESHOLD = 0.002  # RMS threshold for sound detection
Y_LIMIT = 0.05  # Fixed y-axis limit in volts (50 mV)

# Queue for audio data
audio_queue = queue.Queue(maxsize=10)

# Buffer for each channel
channel_buffers = [[] for _ in range(CHANNELS)]

# Sound detection state
sound_detected = False


def detect_sound(signal, threshold=SOUND_THRESHOLD):
    """
    Detect if there's a sound event in the signal.

    Args:
        signal: Audio signal (1D or 2D array)
        threshold: RMS threshold for sound detection

    Returns:
        bool: True if sound detected, False otherwise
    """
    rms = np.sqrt(np.mean(signal**2))
    return rms > threshold


def get_subplot_position(channel):
    """
    Map channel number to subplot position in 4x4 grid.

    Grid layout matches physical microphone array:
    Row 3 (top):    MIC15  MIC13  MIC11  MIC09
    Row 2:          MIC16  MIC14  MIC12  MIC10
    Row 1:          MIC01  MIC03  MIC05  MIC07
    Row 0 (bottom): MIC02  MIC04  MIC06  MIC08

    Channel mapping: channel 0 = MIC01, channel 1 = MIC02, etc.

    Returns:
        (row, col) position in the 4x4 grid (0-indexed from bottom-left)
    """
    # Physical layout of the array (channel numbers)
    # Grid positions: [row][col] where row 0 is bottom, row 3 is top
    layout = [
        # Row 0 (bottom)
        [1, 0, 15, 14],      # MIC02, MIC04, MIC06, MIC08
        # Row 1
        [3, 2, 13, 12],      # MIC01, MIC03, MIC05, MIC07
        # Row 2
        [5, 4, 11, 10],   # MIC16, MIC14, MIC12, MIC10
        # Row 3 (top)
        [7, 6, 9, 8]    # MIC15, MIC13, MIC11, MIC09
    ]

    # Find channel in layout
    for row_idx, row in enumerate(layout):
        if channel in row:
            col_idx = row.index(channel)
            return (row_idx, col_idx)

    return (0, 0)  # Fallback


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
    global channel_buffers, sound_detected

    # Get all available audio chunks from queue (limit to prevent lag)
    new_chunks = []
    chunk_count = 0
    max_chunks_per_update = 5  # Limit chunks processed per frame
    try:
        while chunk_count < max_chunks_per_update:
            chunk = audio_queue.get_nowait()
            new_chunks.append(chunk)
            chunk_count += 1
    except queue.Empty:
        pass

    # Process all new chunks (removed sound detection threshold)
    if len(new_chunks) > 0:
        combined_chunk = np.vstack(new_chunks)
        sound_detected = detect_sound(combined_chunk)

        # Always add to buffer (show all audio, not just when above threshold)
        for chunk in new_chunks:
            for ch in range(CHANNELS):
                channel_buffers[ch].append(chunk[:, ch])

    # Keep buffer size manageable
    max_buffer_samples = int(SAMPLE_RATE * DISPLAY_DURATION)
    for ch in range(CHANNELS):
        if len(channel_buffers[ch]) > 0:
            # Concatenate buffer
            channel_data = np.concatenate(channel_buffers[ch])

            # Keep only the most recent samples
            if len(channel_data) > max_buffer_samples:
                channel_data = channel_data[-max_buffer_samples:]
                channel_buffers[ch] = [channel_data]

    # Update each subplot
    for ch in range(CHANNELS):
        row, col = get_subplot_position(ch)
        ax = axes[3 - row, col]  # Flip row for matplotlib (0 at top)

        # Update line data instead of clearing (more efficient)
        if len(channel_buffers[ch]) > 0:
            # Get data for this channel
            data = np.concatenate(channel_buffers[ch])

            # Time axis in milliseconds
            time_axis = np.arange(len(data)) / SAMPLE_RATE * 1000

            # Update or create line
            if len(ax.lines) > 0:
                ax.lines[0].set_data(time_axis, data)
            else:
                ax.plot(time_axis, data, 'b-', linewidth=0.5)

    # Don't update titles - they cause GUI freezing
    return []  # Return empty list for blitting compatibility


if __name__ == "__main__":
    print("=" * 80)
    print("16-CHANNEL MICROPHONE ARRAY VISUALIZATION")
    print("=" * 80)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS}")
    print(f"Display duration: {DISPLAY_DURATION * 1000:.0f} ms")
    print(f"Sound threshold: {SOUND_THRESHOLD} (RMS)")
    print(f"Y-axis range: ±{Y_LIMIT*1000:.1f} mV")
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

    print(f"\nStarting real-time visualization...")

    # Start audio stream
    stream = sd.InputStream(
        device=minidsp_device,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        callback=audio_callback,
        blocksize=BLOCKSIZE
    )

    stream.start()

    # Set up matplotlib figure with 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))

    # Initialize all subplots before starting animation
    print("Initializing plots...")
    for ch in range(CHANNELS):
        row, col = get_subplot_position(ch)
        ax = axes[3 - row, col]  # Flip row for matplotlib (0 at top)

        # Pre-configure each subplot
        ax.set_xlim(0, DISPLAY_DURATION * 1000)
        ax.set_ylim(-Y_LIMIT, Y_LIMIT)
        ax.set_title(f'MIC{ch+1:02d} (Ch{ch})', fontsize=8, weight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=6)

        # Only show x-label on bottom row
        if row == 0:
            ax.set_xlabel('Time (ms)', fontsize=7)

        # Only show y-label on leftmost column
        if col == 0:
            ax.set_ylabel('Amplitude (V)', fontsize=7)

    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Create animation (interval=20ms for ~50 FPS)
    ani = animation.FuncAnimation(
        fig, update_plot, interval=20, cache_frame_data=False, blit=False
    )

    try:
        print("Visualization active. Close the plot window to stop.")
        print("=" * 70)
        plt.show()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop()
        stream.close()
        print("Stream closed.")