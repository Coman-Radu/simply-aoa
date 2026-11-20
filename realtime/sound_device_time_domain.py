import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import numpy as np
from scipy.signal import correlate
import os   

#print(os.getcwd())

# Constants for DOA calculation
SPEED_OF_SOUND = 343.0  # m/s at room temperature
MIC_ARRAY_RADIUS = 0.045  # meters (adjust to your actual array radius)

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


def detect_sound(pdm_signal, threshold=0.02):
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


def process_audio_sample(recording, fs, pdm_channel, mems_channels, sound_threshold=0.02):
    """
    Process a single audio sample to extract DOA.

    Args:
        recording: Audio recording array
        fs: Sample rate
        pdm_channel: PDM mic channel (used for sound detection only)
        mems_channels: List of MEMS mic channels for DOA calculation
        sound_threshold: RMS threshold for sound detection

    Returns:
        azimuth: Direction of arrival in degrees (None if no sound detected)
        time_delays: Array of time delays for each MEMS mic
        sound_detected: Boolean indicating if sound was detected
    """
    # Normalize signals
    recording = recording / (np.max(np.abs(recording)) + 1e-10)

    pdm_signal = recording[:, pdm_channel]

    # Check if sound is detected using PDM mic
    sound_detected = detect_sound(pdm_signal, threshold=sound_threshold)

    if not sound_detected:
        print("No sound detected (below threshold)")
        return None, None, False

    print("Sound detected! Processing DOA...")

    # Use first MEMS channel as reference for cross-correlation
    ref_channel = mems_channels[0]
    ref_signal = recording[:, ref_channel]

    time_delays = []

    # Mic positions in a circular array (6 mics + 1 reference)
    # Reference mic (index 0) is at position (0, 0)
    # Remaining 6 mics distributed around circle
    mic_positions = []
    for i in range(1, len(mems_channels)):
        angle = 2 * np.pi * (i-1) / (len(mems_channels)-1)
        x = MIC_ARRAY_RADIUS * np.cos(angle)
        y = MIC_ARRAY_RADIUS * np.sin(angle)
        mic_positions.append((x, y))

    # Cross-correlate each MEMS mic (except reference) with the reference MEMS mic
    for mems_ch in mems_channels[1:]:
        mems_signal = recording[:, mems_ch]

        # Cross-correlation
        correlation = correlate(mems_signal, ref_signal, mode='same')
        lags = np.arange(-len(ref_signal)//2, len(ref_signal)//2)

        # Find peak
        peak_idx = np.argmax(np.abs(correlation))
        peak_lag = lags[peak_idx]
        peak_time = peak_lag / fs  # seconds

        time_delays.append(peak_time)

    # Calculate DOA
    azimuth = calculate_doa(np.array(time_delays), mic_positions)

    return azimuth, time_delays, sound_detected


if minidsp_device is None:
    print("Error: miniDSP device not found!")
    print("\nAvailable devices:")
    for i, device in enumerate(devices):
        print(f"  [{i}] {device['name']}")
else:
    # Record from miniDSP using its max available channels
    device_info = devices[minidsp_device]
    fs = 16000  # int(device_info['default_samplerate'])
    channels = int(device_info['max_input_channels'])

    # New channel setup:
    # Channel 0: PDM mic (for sound detection only)
    # Channels 1-7: MEMS mics array (for DOA calculation)
    pdm_channel = 0  # PDM mic for sound detection
    mems_channels = list(range(1, 8))  # MEMS microphones for DOA

    print(f"\nRecording single sample from miniDSP ({fs} Hz, {channels} channels)...")
    print(f"PDM channel (trigger): {pdm_channel}")
    print(f"MEMS channels (DOA): {mems_channels}")

    sample_duration = 0.5  # seconds
    sound_threshold = 0.02  # Adjust this threshold based on your environment

    # Record 0.5 seconds of audio
    recording = sd.rec(int(sample_duration * fs),
                     samplerate=fs,
                     channels=channels,
                     dtype='float32',
                     device=minidsp_device)
    sd.wait()

    # Process the audio sample
    azimuth, time_delays, sound_detected = process_audio_sample(
        recording, fs, pdm_channel, mems_channels, sound_threshold
    )

    if sound_detected and azimuth is not None:
        # Print results
        print(f"\nDOA: {azimuth:.1f}°")
        print("Time delays (μs):")
        # Note: time_delays correspond to mems_channels[1:] (channels 2-7)
        for i, td in enumerate(time_delays):
            print(f"  M{mems_channels[i+1]} vs M{mems_channels[0]}: {td*1e6:.1f}")

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Polar plot for DOA
        ax1 = plt.subplot(1, 2, 1, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_title('Direction of Arrival (Azimuth)')

        # Plot DOA as arrow
        azimuth_rad = np.radians(azimuth)
        ax1.arrow(azimuth_rad, 0, 0, 0.8,
                 head_width=0.0, head_length=0.1,
                 fc='red', ec='red', linewidth=2)
        ax1.set_ylim(0, 1)

        # Time delays plot - use channels 2-7 for x-axis labels
        mems_comparison_channels = mems_channels[1:]
        ax2.bar(range(len(time_delays)), np.array(time_delays) * 1e6)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax2.set_xticks(range(len(time_delays)))
        ax2.set_xticklabels([f'M{ch}' for ch in mems_comparison_channels])
        ax2.set_xlabel(f'MEMS Channel (vs M{mems_channels[0]} reference)')
        ax2.set_ylabel('Time Delay (μs)')
        ax2.set_title('Time Delays Between MEMS Mics')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
    elif not sound_detected:
        print("\nNo sound detected - adjust threshold or make louder sound.")
    else:
        print("\nCould not calculate DOA from the recording.")
