import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write
import os

print(os.getcwd())

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
else:
    # Record from miniDSP using its max available channels
    device_info = devices[minidsp_device]
    fs = 16000 #int(device_info['default_samplerate'])
    channels = int(device_info['max_input_channels'])

    print(f"\nRecording 1 second from miniDSP ({fs} Hz, {channels} channels)...")

    try:
        recording = sd.rec(int(fs), samplerate=fs, channels=channels,
                           dtype='float32', device=minidsp_device)
    
        sd.wait()
        print(f"✓ Recording successful! Shape: {recording.shape}")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        
time_record = np.arange(len(recording))/fs


write("example.wav", fs, recording.astype(np.float32))

sd.play(recording[:,0], fs)

plt.plot(time_record, recording)
plt.show()