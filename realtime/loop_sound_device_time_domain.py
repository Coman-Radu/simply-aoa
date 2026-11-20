import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
import os
import time

print(os.getcwd())

# Constants for DOA calculation
SPEED_OF_SOUND = 343.0  # m/s at room temperature
MIC_ARRAY_RADIUS = 0.09  # meters (adjust to your actual array radius)

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
        time_delays: Array of time delays (in seconds) relative to center mic
        mic_positions: Array of mic positions in 2D [(x, y), ...] in meters

    Returns:
        angle: Azimuth angle in degrees (0° is North, clockwise)
    """
    # Use weighted average of angle estimates from each microphone
    angles = []
    weights = []

    for i, (td, pos) in enumerate(zip(time_delays, mic_positions)):
        # Calculate distance difference
        distance_diff = td * SPEED_OF_SOUND

        # Get mic angle from center
        mic_angle = np.arctan2(pos[1], pos[0])

        # The time delay tells us if sound arrives early (positive) or late (negative)
        # at this mic compared to center
        # If positive delay, source is on opposite side of mic
        if abs(distance_diff) < MIC_ARRAY_RADIUS:
            # Use cosine rule: d = r * cos(theta)
            # where theta is angle between source direction and mic direction
            cos_theta = distance_diff / MIC_ARRAY_RADIUS
            cos_theta = np.clip(cos_theta, -1, 1)  # Ensure valid range

            # Source angle relative to mic
            # Positive delay means source is opposite to mic
            source_angle = mic_angle + np.pi - np.arccos(cos_theta)

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


def process_audio_sample(recording, fs, pdm_channel, mems_channels):
    """
    Process a single audio sample to extract DOA.

    Returns:
        azimuth: Direction of arrival in degrees
        time_delays: Array of time delays for each MEMS mic
    """
    # Normalize signals
    recording = recording / (np.max(np.abs(recording)) + 1e-10)

    pdm_signal = recording[:, pdm_channel]
    time_delays = []

    # Mic positions in a circular array (7 mics)
    # Assuming uniform distribution around circle
    mic_positions = []
    for i in range(7):
        angle = 2 * np.pi * i / 7
        x = MIC_ARRAY_RADIUS * np.cos(angle)
        y = MIC_ARRAY_RADIUS * np.sin(angle)
        mic_positions.append((x, y))

    # Cross-correlate each MEMS mic with the PDM center mic
    for mems_ch in mems_channels:
        mems_signal = recording[:, mems_ch]

        # Cross-correlation
        correlation = correlate(mems_signal, pdm_signal, mode='same')
        lags = np.arange(-len(pdm_signal)//2, len(pdm_signal)//2)

        # Find peak
        peak_idx = np.argmax(np.abs(correlation))
        peak_lag = lags[peak_idx]
        peak_time = peak_lag / fs  # seconds

        time_delays.append(peak_time)

    # Calculate DOA
    azimuth = calculate_doa(np.array(time_delays), mic_positions)

    return azimuth, time_delays


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

    # Circular array setup: 7 MEMS mics (channels 0-6) + 1 PDM center mic (channel 7)
    pdm_channel = 7  # Center reference microphone
    mems_channels = list(range(7))  # Peripheral MEMS microphones

    print(f"\nStarting continuous recording from miniDSP ({fs} Hz, {channels} channels)...")
    print("Press Ctrl+C to stop\n")

    sample_duration = 0.5  # seconds per sample
    sample_interval = 1.0  # process a sample every second

    try:
        sample_count = 0
        while True:
            start_time = time.time()

            # Record 0.5 seconds of audio
            recording = sd.rec(int(sample_duration * fs),
                             samplerate=fs,
                             channels=channels,
                             dtype='float32',
                             device=minidsp_device)
            sd.wait()

            # Process the audio sample
            azimuth, time_delays = process_audio_sample(
                recording, fs, pdm_channel, mems_channels
            )

            sample_count += 1

            if azimuth is not None:
                # Print results
                print(f"[Sample {sample_count}] DOA: {azimuth:.1f}°", end="  |  ")
                print("Time delays (μs): ", end="")
                for i, td in enumerate(time_delays):
                    print(f"M{i}:{td*1e6:+6.1f} ", end="")
                print()
            else:
                print(f"[Sample {sample_count}] Could not calculate DOA")

            # Wait for the remaining time to maintain 1-second interval
            elapsed = time.time() - start_time
            wait_time = sample_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

    except KeyboardInterrupt:
        print(f"\n\nStopped after {sample_count} samples.")
