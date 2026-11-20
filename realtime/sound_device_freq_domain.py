import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftshift
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
    fs = 16000  # int(device_info['default_samplerate'])
    channels = int(device_info['max_input_channels'])

    print(f"\nRecording 1 second from miniDSP ({fs} Hz, {channels} channels)...")

    try:
        recording = sd.rec(int(fs), samplerate=fs, channels=channels,
                           dtype='float32', device=minidsp_device)

        sd.wait()
        print(f"Recording successful! Shape: {recording.shape}")

    except Exception as e:
        print(f"Error: {e}")

    time_record = np.arange(len(recording))/fs

    # Circular array setup: 7 MEMS mics (channels 0-6) + 1 PDM center mic (channel 7)
    # Normalize signals
    recording = recording / np.max(np.abs(recording))

    pdm_channel = 7  # Center reference microphone
    mems_channels = list(range(7))  # Peripheral MEMS microphones

    pdm_signal = recording[:, pdm_channel]
    N = len(pdm_signal)

    # Cross-correlate each MEMS mic with the PDM center mic using frequency domain
    print("\nFrequency-domain cross-correlation results (MEMS-PDM pairs):")

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, mems_ch in enumerate(mems_channels):
        mems_signal = recording[:, mems_ch]

        # Frequency domain cross-correlation
        N_fft = 2 * N - 1
        freq_mems = fft(mems_signal, N_fft)
        freq_pdm = fft(pdm_signal, N_fft)

        # Cross-power spectrum with normalization (GCC-PHAT)
        cross_spectrum = np.conjugate(freq_pdm) * freq_mems
        cross_spectrum_norm = cross_spectrum / (np.abs(cross_spectrum) + 1e-10)

        # Inverse FFT to get correlation
        correlation = ifft(cross_spectrum_norm)
        correlation = fftshift(correlation)

        lags = np.arange(-N_fft//2, N_fft//2)
        time_lags = lags / fs * 1e6  # Convert to microseconds

        # Find peak
        peak_idx = np.argmax(correlation.real)
        peak_lag = lags[peak_idx]
        peak_time = peak_lag / fs * 1e6  # microseconds

        print(f"  MEMS {mems_ch} <-> PDM: lag = {peak_lag} samples ({peak_time:.2f} us)")

        # Plot
        axes[idx].plot(time_lags[:-1], correlation.real[:-1])
        axes[idx].axvline(peak_time, color='r', linestyle='--', label=f'Peak: {peak_time:.2f} us')
        axes[idx].set_title(f'MEMS {mems_ch} <-> PDM (Freq Domain)')
        axes[idx].set_xlabel('Time lag (us)')
        axes[idx].set_ylabel('Correlation')
        axes[idx].legend()
        axes[idx].grid(True)

    # Remove extra subplot
    fig.delaxes(axes[7])

    plt.tight_layout()
    plt.show()

    # Optional: Plot time-domain signals
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_record, pdm_signal, label='PDM (center)', linewidth=2, alpha=0.7)
    for mems_ch in mems_channels:
        ax.plot(time_record, recording[:, mems_ch], label=f'MEMS {mems_ch}', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('8-Channel Microphone Array Signals')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
