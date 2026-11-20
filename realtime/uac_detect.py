"""
USB Audio Class 2.0 - 8 Channel Audio Recorder
Records 8 channels of audio from a single USB audio device
"""

import sounddevice as sd
import numpy as np
import wave
import sys
from datetime import datetime
import matplotlib.pyplot as plt

# Configuration
SAMPLE_RATE = 48000  # Hz
CHANNELS = 8
DURATION = 10  # seconds
DTYPE = np.int16
BLOCKSIZE = 4096

def list_audio_devices():
    """List all available audio devices"""
    print("\nAvailable Audio Devices:")
    print("=" * 80)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] >= CHANNELS:
            print(f"[{i}] {device['name']}")
            print(f"    Max Input Channels: {device['max_input_channels']}")
            print(f"    Default Sample Rate: {device['default_samplerate']}")
            print()
    return devices

def select_device(devices):
    """Select appropriate USB audio device with 8 channels"""
    # Try to find a device with exactly 8 or more input channels
    suitable_devices = [
        (i, dev) for i, dev in enumerate(devices)
        if dev['max_input_channels'] >= CHANNELS
    ]

    if not suitable_devices:
        print(f"Error: No audio device found with {CHANNELS} or more input channels")
        sys.exit(1)

    if len(suitable_devices) == 1:
        device_id, device = suitable_devices[0]
        print(f"\nAuto-selected device [{device_id}]: {device['name']}")
        return device_id

    # Multiple devices available, let user choose
    print(f"\nMultiple devices with {CHANNELS}+ channels found.")
    while True:
        try:
            device_id = int(input("Enter device ID: "))
            if any(dev_id == device_id for dev_id, _ in suitable_devices):
                return device_id
            else:
                print(f"Please select a device with at least {CHANNELS} input channels")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)

def record_audio(device_id, duration=DURATION):
    """Record audio from the selected device"""
    print(f"\nRecording {CHANNELS} channels for {duration} seconds...")
    print("Recording started...")

    try:
        # Record audio
        recording = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=device_id,
            blocking=True
        )

        print("Recording completed!")
        return recording

    except Exception as e:
        print(f"Error during recording: {e}")
        sys.exit(1)

def save_audio(recording, filename=None):
    """Save recorded audio to WAV file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"uac_recording_{CHANNELS}ch_{timestamp}.wav"

    print(f"\nSaving to: {filename}")

    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(recording.tobytes())

        print(f"Successfully saved {filename}")
        print(f"Duration: {len(recording) / SAMPLE_RATE:.2f} seconds")
        print(f"Channels: {CHANNELS}")
        print(f"Sample Rate: {SAMPLE_RATE} Hz")

    except Exception as e:
        print(f"Error saving file: {e}")
        sys.exit(1)

def main():
    """Main function"""
    print("=" * 80)
    print("USB Audio Class 2.0 - 8 Channel Recorder")
    print("=" * 80)

    # List and select device
    devices = list_audio_devices()
    device_id = select_device(devices)

    # Get recording duration from user (optional)
    try:
        duration_input = input(f"\nEnter recording duration in seconds (default: {DURATION}): ").strip()
        duration = int(duration_input) if duration_input else DURATION
    except ValueError:
        duration = DURATION
        print(f"Using default duration: {duration} seconds")

    # Record audio
    recording = record_audio(device_id, duration)
    
    plot_time = np.arange(np.size(recording, 0))/SAMPLE_RATE
     
    plt.figure()
    plt.plot(plot_time, recording)
    #plt.xlabel('time (s)')
    #plt.ylabel('amplitude')
    plt.show()
    # Save audio
    save_audio(recording)

    print("\nDone!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nRecording cancelled by user")
        sys.exit(0)
