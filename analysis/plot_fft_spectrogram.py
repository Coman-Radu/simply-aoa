#!/usr/bin/env python3
"""
FFT and Spectrogram Plot for Binary Drone Audio
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from scipy.signal import spectrogram


def main():
    # Load the WAV file
    wav_file = "Binary_Drone_Audio_yes_drone_B_S2_D1_067-bebop_000_.wav"
    print(f"Loading: {wav_file}")

    sample_rate, audio = wavfile.read(wav_file)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # Normalize to [-1, 1]
    audio = audio.astype(np.float64)
    audio = audio / np.max(np.abs(audio))

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(audio) / sample_rate:.2f} seconds")

    # Compute FFT
    fft_values = rfft(audio)
    freqs = rfftfreq(len(audio), 1 / sample_rate)
    magnitude_db = 20 * np.log10(np.abs(fft_values) + 1e-10)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: FFT Magnitude Spectrum
    ax1.plot(freqs, magnitude_db, 'b-', linewidth=0.5)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Fourier Transform (FFT Magnitude Spectrum)')
    ax1.set_xlim([0, sample_rate / 2])
    ax1.set_ylim([np.max(magnitude_db) - 80, np.max(magnitude_db) + 5])
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spectrogram
    f, t, Sxx = spectrogram(audio, sample_rate, nperseg=2048, noverlap=1536)
    im = ax2.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10),
                        shading='gouraud', cmap='viridis', vmin=-80, vmax=-20)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Spectrogram')
    ax2.set_ylim([0, 10000])
    plt.colorbar(im, ax=ax2, label='Power (dB)')

    plt.suptitle(f'Analysis: {wav_file}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fft_spectrogram_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: fft_spectrogram_analysis.png")
    plt.show()


if __name__ == "__main__":
    main()
