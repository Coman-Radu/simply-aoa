#!/usr/bin/env python3
"""
Drone Audio FFT Analysis
========================

Loads the binary drone audio file and plots:
1. Time-domain waveform
2. Frequency spectrum (FFT magnitude)
3. Power spectral density
4. Spectrogram (time-frequency representation)

This helps identify the dominant frequency components in the drone propeller noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import spectrogram, welch
import glob


def load_drone_audio():
    """Load the Binary drone audio file"""
    wav_files = glob.glob("Binary*.wav")
    if not wav_files:
        raise FileNotFoundError("No Binary*.wav file found in current directory")

    wav_file = wav_files[0]
    print(f"Loading audio from: {wav_file}")

    sample_rate, audio = wavfile.read(wav_file)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio[:, 0]
        print("Converted stereo to mono (channel 0)")

    # Normalize to [-1, 1]
    audio = audio.astype(np.float64)
    audio = audio / np.max(np.abs(audio))

    duration = len(audio) / sample_rate

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Number of samples: {len(audio)}")
    print(f"Amplitude range: [{np.min(audio):.3f}, {np.max(audio):.3f}]")

    return audio, sample_rate, wav_file


def compute_fft(audio, sample_rate):
    """Compute FFT and return frequency and magnitude"""
    # Use rfft for real-valued signal (only positive frequencies)
    fft_values = rfft(audio)
    freqs = rfftfreq(len(audio), 1/sample_rate)

    # Magnitude spectrum (in linear scale)
    magnitude = np.abs(fft_values)

    # Power spectrum (magnitude squared)
    power = magnitude**2

    # Convert to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-10)  # Add small value to avoid log(0)
    power_db = 10 * np.log10(power + 1e-10)

    return freqs, magnitude, magnitude_db, power, power_db


def find_dominant_frequencies(freqs, magnitude, num_peaks=10):
    """Find the dominant frequency components"""
    # Find peaks
    peak_indices = np.argsort(magnitude)[::-1][:num_peaks]
    peak_indices = peak_indices[np.argsort(freqs[peak_indices])]  # Sort by frequency

    peak_freqs = freqs[peak_indices]
    peak_mags = magnitude[peak_indices]

    return peak_freqs, peak_mags


def plot_analysis(audio, sample_rate, freqs, magnitude, magnitude_db, power_db, wav_file):
    """Create comprehensive frequency analysis plots"""
    fig = plt.figure(figsize=(16, 10))

    # Time vector
    t = np.arange(len(audio)) / sample_rate

    # === Panel 1: Time-domain waveform (full) ===
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(t, audio, 'b-', linewidth=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Time-Domain Waveform (Full)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, t[-1]])

    # === Panel 2: Time-domain waveform (first 0.1s) ===
    ax2 = plt.subplot(3, 3, 2)
    t_zoom = t[:int(0.1 * sample_rate)]
    audio_zoom = audio[:int(0.1 * sample_rate)]
    ax2.plot(t_zoom, audio_zoom, 'b-', linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Waveform (First 100 ms)')
    ax2.grid(True, alpha=0.3)

    # === Panel 3: Amplitude histogram ===
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(audio, bins=100, color='blue', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Amplitude')
    ax3.set_ylabel('Count')
    ax3.set_title('Amplitude Distribution')
    ax3.grid(True, alpha=0.3)

    # === Panel 4: FFT Magnitude (Linear scale, full range) ===
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(freqs, magnitude, 'b-', linewidth=0.5)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude (Linear)')
    ax4.set_title('FFT Magnitude Spectrum (Linear Scale)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, sample_rate/2])

    # === Panel 5: FFT Magnitude (dB scale, full range) ===
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(freqs, magnitude_db, 'b-', linewidth=0.5)
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Magnitude (dB)')
    ax5.set_title('FFT Magnitude Spectrum (dB Scale)')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, sample_rate/2])
    ax5.set_ylim([np.max(magnitude_db) - 80, np.max(magnitude_db) + 5])

    # === Panel 6: FFT Magnitude (dB scale, zoomed to 0-5000 Hz) ===
    ax6 = plt.subplot(3, 3, 6)
    mask = freqs <= 5000
    ax6.plot(freqs[mask], magnitude_db[mask], 'b-', linewidth=0.8)
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Magnitude (dB)')
    ax6.set_title('FFT Magnitude (0-5000 Hz)')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0, 5000])

    # Find and mark dominant frequencies in this range
    peak_freqs, peak_mags = find_dominant_frequencies(freqs[mask], magnitude[mask], num_peaks=5)
    for pf, pm in zip(peak_freqs, peak_mags):
        pm_db = 20 * np.log10(pm + 1e-10)
        ax6.plot(pf, pm_db, 'ro', markersize=8)
        ax6.annotate(f'{pf:.0f} Hz', xy=(pf, pm_db), xytext=(10, 10),
                    textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # === Panel 7: Power Spectral Density (Welch method) ===
    ax7 = plt.subplot(3, 3, 7)
    freqs_psd, psd = welch(audio, sample_rate, nperseg=2048)
    psd_db = 10 * np.log10(psd + 1e-10)
    ax7.plot(freqs_psd, psd_db, 'g-', linewidth=1)
    ax7.set_xlabel('Frequency (Hz)')
    ax7.set_ylabel('PSD (dB/Hz)')
    ax7.set_title('Power Spectral Density (Welch Method)')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([0, sample_rate/2])

    # === Panel 8: Spectrogram ===
    ax8 = plt.subplot(3, 3, 8)
    f, t_spec, Sxx = spectrogram(audio, sample_rate, nperseg=2048, noverlap=1536)
    im = ax8.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                       shading='gouraud', cmap='viridis', vmin=-80, vmax=-20)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Frequency (Hz)')
    ax8.set_title('Spectrogram')
    ax8.set_ylim([0, 8000])
    plt.colorbar(im, ax=ax8, label='Power (dB)')

    # === Panel 9: Cumulative energy spectrum ===
    ax9 = plt.subplot(3, 3, 9)
    power_linear = magnitude**2
    cumulative_energy = np.cumsum(power_linear)
    cumulative_energy = cumulative_energy / cumulative_energy[-1] * 100  # Normalize to percentage
    ax9.plot(freqs, cumulative_energy, 'r-', linewidth=2)
    ax9.set_xlabel('Frequency (Hz)')
    ax9.set_ylabel('Cumulative Energy (%)')
    ax9.set_title('Cumulative Energy Spectrum')
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim([0, sample_rate/2])
    ax9.axhline(50, color='k', linestyle='--', alpha=0.5, label='50%')
    ax9.axhline(90, color='k', linestyle='--', alpha=0.5, label='90%')
    ax9.legend()

    # Find frequency where 50% and 90% of energy is contained
    f_50 = freqs[np.argmin(np.abs(cumulative_energy - 50))]
    f_90 = freqs[np.argmin(np.abs(cumulative_energy - 90))]
    ax9.axvline(f_50, color='b', linestyle=':', alpha=0.7)
    ax9.axvline(f_90, color='b', linestyle=':', alpha=0.7)
    ax9.text(f_50, 55, f'{f_50:.0f} Hz', rotation=90, verticalalignment='bottom')
    ax9.text(f_90, 92, f'{f_90:.0f} Hz', rotation=90, verticalalignment='bottom')

    plt.suptitle(f'Drone Audio FFT Analysis: {wav_file}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('drone_audio_fft_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved: drone_audio_fft_analysis.png")


def print_statistics(audio, sample_rate, freqs, magnitude):
    """Print statistical analysis"""
    print("\n" + "="*60)
    print("FREQUENCY DOMAIN STATISTICS")
    print("="*60)

    # Time domain stats
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    crest_factor = peak / rms

    print(f"\nTime Domain:")
    print(f"  RMS amplitude: {rms:.4f}")
    print(f"  Peak amplitude: {peak:.4f}")
    print(f"  Crest factor: {crest_factor:.2f}")

    # Frequency domain stats
    power = magnitude**2
    total_power = np.sum(power)

    # Find dominant frequencies
    peak_freqs, peak_mags = find_dominant_frequencies(freqs, magnitude, num_peaks=10)

    print(f"\nTop 10 Dominant Frequencies:")
    for i, (pf, pm) in enumerate(zip(peak_freqs, peak_mags), 1):
        power_pct = (pm**2 / total_power) * 100
        print(f"  {i:2d}. {pf:7.1f} Hz - Magnitude: {pm:10.1f} ({power_pct:5.2f}% of total power)")

    # Energy distribution
    cumulative_energy = np.cumsum(power) / total_power * 100
    f_50 = freqs[np.argmin(np.abs(cumulative_energy - 50))]
    f_90 = freqs[np.argmin(np.abs(cumulative_energy - 90))]
    f_99 = freqs[np.argmin(np.abs(cumulative_energy - 99))]

    print(f"\nEnergy Distribution:")
    print(f"  50% of energy below: {f_50:.0f} Hz")
    print(f"  90% of energy below: {f_90:.0f} Hz")
    print(f"  99% of energy below: {f_99:.0f} Hz")

    # Bandwidth estimates
    mean_freq = np.sum(freqs * power) / total_power
    variance = np.sum(((freqs - mean_freq)**2) * power) / total_power
    std_freq = np.sqrt(variance)

    print(f"\nBandwidth:")
    print(f"  Mean frequency: {mean_freq:.1f} Hz")
    print(f"  Std deviation: {std_freq:.1f} Hz")
    print(f"  Bandwidth (±1 std): {mean_freq - std_freq:.1f} - {mean_freq + std_freq:.1f} Hz")

    print("="*60)


def main():
    print("="*60)
    print("DRONE AUDIO FFT ANALYSIS")
    print("="*60)

    # Load audio
    audio, sample_rate, wav_file = load_drone_audio()

    # Compute FFT
    print("\nComputing FFT...")
    freqs, magnitude, magnitude_db, power, power_db = compute_fft(audio, sample_rate)
    print(f"Frequency resolution: {freqs[1] - freqs[0]:.2f} Hz")

    # Print statistics
    print_statistics(audio, sample_rate, freqs, magnitude)

    # Create plots
    print("\nGenerating plots...")
    plot_analysis(audio, sample_rate, freqs, magnitude, magnitude_db, power_db, wav_file)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
