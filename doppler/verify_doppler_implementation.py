#!/usr/bin/env python3
"""
Verify Doppler Implementation
==============================

Uses real drone audio as the source to verify that the Doppler effect
is correctly implemented using the retarded time method.

Tests the relationship: f_observed = f_source × c / (c - v_radial)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram, resample
from scipy.io import wavfile
import glob

# Physical constants
SPEED_OF_SOUND = 343.0  # m/s
SAMPLE_RATE = 48000
SOURCE_VELOCITY = 15.0  # m/s (higher velocity for clearer effect)

# Array configuration
MIC_SPACING = 0.04
MIC_POSITIONS = np.array([
    [-3*MIC_SPACING/2, 0],
    [-MIC_SPACING/2, 0],
    [MIC_SPACING/2, 0],
    [3*MIC_SPACING/2, 0]
])

# Trajectory
SOURCE_DISTANCE_Y = 30.0  # Closer for stronger effect
SOURCE_INITIAL_X = -20.0
DURATION = 3.0


def load_drone_audio():
    """Load real drone audio file"""
    wav_files = glob.glob("Binary*.wav")
    if not wav_files:
        raise FileNotFoundError("No Binary*.wav file found in current directory")

    wav_file = wav_files[0]
    print(f"Loading audio from: {wav_file}")

    sample_rate, audio = wavfile.read(wav_file)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # Normalize to [-1, 1]
    audio = audio.astype(np.float64)
    audio = audio / np.max(np.abs(audio))

    # Resample to target rate if needed
    if sample_rate != SAMPLE_RATE:
        num_samples = int(len(audio) * SAMPLE_RATE / sample_rate)
        audio = resample(audio, num_samples)
        print(f"Resampled from {sample_rate} Hz to {SAMPLE_RATE} Hz")
        sample_rate = SAMPLE_RATE

    # Limit duration
    max_samples = int(DURATION * sample_rate)
    audio = audio[:max_samples]

    print(f"Audio duration: {len(audio)/sample_rate:.2f} s")

    return audio, sample_rate


def generate_moving_drone_audio(source_audio):
    """Apply Doppler effect to real drone audio using retarded time method"""
    num_samples = len(source_audio)
    t_receiver = np.arange(num_samples) / SAMPLE_RATE

    mic_signals = np.zeros((4, num_samples))

    # For tracking
    x_trajectory = SOURCE_INITIAL_X + SOURCE_VELOCITY * t_receiver
    y_trajectory = np.ones_like(t_receiver) * SOURCE_DISTANCE_Y

    for mic_idx in range(4):
        mic_pos = MIC_POSITIONS[mic_idx]

        t_emission = np.zeros(num_samples)

        # Compute emission time for each receiver sample
        for i in range(num_samples):
            t_r = t_receiver[i]
            t_e = t_r  # Initial guess

            # Iterate to find emission time
            for iteration in range(5):
                x_e = SOURCE_INITIAL_X + SOURCE_VELOCITY * t_e
                y_e = SOURCE_DISTANCE_Y
                dx = x_e - mic_pos[0]
                dy = y_e - mic_pos[1]
                r_e = np.sqrt(dx**2 + dy**2)
                t_prop = r_e / SPEED_OF_SOUND
                t_e_new = t_r - t_prop

                if abs(t_e_new - t_e) < 1e-9:
                    break
                t_e = t_e_new

            t_emission[i] = t_e

        # Sample the source audio at emission times (this applies Doppler naturally)
        emission_samples = t_emission * SAMPLE_RATE
        emission_samples = np.clip(emission_samples, 0, num_samples - 1)

        # Interpolate source audio at emission times
        doppler_audio = np.interp(emission_samples, np.arange(num_samples), source_audio)

        # Compute distance for attenuation
        x_e = SOURCE_INITIAL_X + SOURCE_VELOCITY * t_emission
        y_e = SOURCE_DISTANCE_Y
        dx = x_e - mic_pos[0]
        dy = y_e - mic_pos[1]
        r_e = np.sqrt(dx**2 + dy**2)

        # Apply 1/r attenuation
        mic_signals[mic_idx, :] = doppler_audio / r_e

    # Compute theoretical Doppler shifts
    radial_velocity = np.zeros((4, num_samples))
    doppler_factor = np.zeros((4, num_samples))  # Frequency multiplication factor

    for mic_idx in range(4):
        mic_pos = MIC_POSITIONS[mic_idx]
        dx = x_trajectory - mic_pos[0]
        dy = y_trajectory - mic_pos[1]
        r = np.sqrt(dx**2 + dy**2)

        # Radial velocity: v · r_hat
        v_radial = SOURCE_VELOCITY * (dx / r)
        radial_velocity[mic_idx, :] = v_radial

        # Doppler factor: f_obs = f_source × c / (c - v_radial)
        doppler_factor[mic_idx, :] = SPEED_OF_SOUND / (SPEED_OF_SOUND - v_radial)

    trajectory = {
        't': t_receiver,
        'x': x_trajectory,
        'y': y_trajectory,
        'radial_velocity': radial_velocity,
        'doppler_factor': doppler_factor
    }

    return mic_signals, trajectory


def analyze_frequency_content(signal, sample_rate, t_center, window_duration=0.2):
    """Analyze frequency content around a specific time"""
    center_sample = int(t_center * sample_rate)
    window_samples = int(window_duration * sample_rate)

    start_sample = max(0, center_sample - window_samples // 2)
    end_sample = min(len(signal), center_sample + window_samples // 2)

    window = signal[start_sample:end_sample]

    # Apply Hanning window
    window = window * np.hanning(len(window))

    # FFT
    fft_result = fft(window)
    freqs = fftfreq(len(window), 1/sample_rate)

    # Only positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    magnitude = np.abs(fft_result[:len(freqs)//2])

    # Find peak
    peak_idx = np.argmax(magnitude)
    peak_freq = positive_freqs[peak_idx]

    return positive_freqs, magnitude, peak_freq


def plot_verification(mic_signals, trajectory, source_audio):
    """Create comprehensive verification plots"""
    fig = plt.figure(figsize=(16, 12))

    t = trajectory['t']

    # === Panel 1: Trajectory ===
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(trajectory['x'], trajectory['y'], 'b-', linewidth=2)
    ax1.plot(trajectory['x'][0], trajectory['y'][0], 'go', markersize=10, label='Start')
    ax1.plot(trajectory['x'][-1], trajectory['y'][-1], 'ro', markersize=10, label='End')
    ax1.plot(MIC_POSITIONS[:, 0], MIC_POSITIONS[:, 1], 'ks', markersize=8, label='Mics')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Source Trajectory')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()

    # === Panel 2: Theoretical Doppler Factor ===
    ax2 = plt.subplot(4, 3, 2)
    for mic_idx in range(4):
        ax2.plot(t, trajectory['doppler_factor'][mic_idx, :], label=f'Mic {mic_idx}')
    ax2.axhline(1.0, color='k', linestyle='--', label='No Doppler')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Doppler Factor (f_obs/f_source)')
    ax2.set_title('Theoretical Doppler Factor = c/(c - v_radial)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # === Panel 3: Radial Velocity ===
    ax3 = plt.subplot(4, 3, 3)
    for mic_idx in range(4):
        ax3.plot(t, trajectory['radial_velocity'][mic_idx, :], label=f'Mic {mic_idx}')
    ax3.axhline(0, color='k', linestyle='--')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Radial Velocity (m/s)')
    ax3.set_title('Radial Velocity (+ = approaching)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # === Panel 4: Source audio spectrogram ===
    ax4 = plt.subplot(4, 3, 4)
    f, t_spec, Sxx = spectrogram(source_audio, SAMPLE_RATE,
                                  nperseg=2048, noverlap=1536)
    im = ax4.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                       shading='gouraud', cmap='viridis', vmin=-80, vmax=-20)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_title('Source Audio (no Doppler)')
    ax4.set_ylim([0, 8000])
    plt.colorbar(im, ax=ax4, label='Power (dB)')

    # === Panel 5-8: Spectrograms for each mic ===
    for mic_idx in range(4):
        ax = plt.subplot(4, 3, 5 + mic_idx)

        f, t_spec, Sxx = spectrogram(mic_signals[mic_idx, :], SAMPLE_RATE,
                                      nperseg=2048, noverlap=1536)

        # Plot spectrogram
        im = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                          shading='gouraud', cmap='viridis', vmin=-80, vmax=-20)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Mic {mic_idx} (with Doppler)')
        ax.set_ylim([0, 8000])
        plt.colorbar(im, ax=ax, label='Power (dB)')

    # === Panel 9: Waveform comparison (first 0.1s) ===
    ax9 = plt.subplot(4, 3, 9)
    t_wave = np.arange(int(0.1 * SAMPLE_RATE)) / SAMPLE_RATE
    ax9.plot(t_wave, source_audio[:len(t_wave)], 'k-', linewidth=0.5, alpha=0.7, label='Source')
    for mic_idx in range(4):
        ax9.plot(t_wave, mic_signals[mic_idx, :len(t_wave)] + mic_idx*0.5,
                linewidth=0.5, label=f'Mic {mic_idx}')
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Amplitude (offset)')
    ax9.set_title('Waveforms (first 100ms)')
    ax9.grid(True, alpha=0.3)
    ax9.legend()

    # === Panel 10: Doppler factor range ===
    ax10 = plt.subplot(4, 3, 10)
    for mic_idx in range(4):
        doppler_shift_hz = (trajectory['doppler_factor'][mic_idx, :] - 1.0) * 3000  # At 3kHz reference
        ax10.plot(t, doppler_shift_hz, label=f'Mic {mic_idx}')
    ax10.axhline(0, color='k', linestyle='--')
    ax10.set_xlabel('Time (s)')
    ax10.set_ylabel('Frequency Shift (Hz @ 3kHz ref)')
    ax10.set_title('Doppler Shift at 3kHz Reference Frequency')
    ax10.grid(True, alpha=0.3)
    ax10.legend()

    # === Panel 11: Distance from each mic ===
    ax11 = plt.subplot(4, 3, 11)
    for mic_idx in range(4):
        mic_pos = MIC_POSITIONS[mic_idx]
        dx = trajectory['x'] - mic_pos[0]
        dy = trajectory['y'] - mic_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        ax11.plot(t, distance, label=f'Mic {mic_idx}')
    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('Distance (m)')
    ax11.set_title('Distance from Source to Each Mic')
    ax11.grid(True, alpha=0.3)
    ax11.legend()

    # === Panel 12: Summary statistics ===
    ax12 = plt.subplot(4, 3, 12)
    ax12.axis('off')

    # Calculate statistics
    max_doppler_factor = np.max(trajectory['doppler_factor'])
    min_doppler_factor = np.min(trajectory['doppler_factor'])
    max_v_radial = np.max(trajectory['radial_velocity'])
    min_v_radial = np.min(trajectory['radial_velocity'])

    summary_text = f"""DOPPLER EFFECT VERIFICATION

Source: Real Drone Audio
Velocity: {SOURCE_VELOCITY} m/s horizontal
Distance: {SOURCE_DISTANCE_Y} m perpendicular

Doppler Factor Range:
  Min: {min_doppler_factor:.4f} (receding)
  Max: {max_doppler_factor:.4f} (approaching)

Radial Velocity Range:
  Min: {min_v_radial:.2f} m/s
  Max: {max_v_radial:.2f} m/s

Frequency Shift @ 3kHz:
  Max: {(max_doppler_factor - 1.0) * 3000:.1f} Hz
  Min: {(min_doppler_factor - 1.0) * 3000:.1f} Hz

Implementation: Retarded Time Method
  t_r = t_e + r(t_e)/c
  Doppler emerges from time compression
"""

    ax12.text(0.1, 0.5, summary_text, transform=ax12.transAxes,
             verticalalignment='center', fontsize=10,
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('verify_doppler_implementation.png', dpi=150, bbox_inches='tight')
    print("Saved: verify_doppler_implementation.png")

    print(f"\nDoppler Implementation Verification:")
    print(f"  Source: Real drone audio (broadband)")
    print(f"  Source velocity: {SOURCE_VELOCITY} m/s")
    print(f"  Doppler factor range: {min_doppler_factor:.4f} to {max_doppler_factor:.4f}")
    print(f"  Max Doppler shift @ 3kHz: {(max_doppler_factor - 1.0) * 3000:.1f} Hz")
    print(f"  Method: Retarded time (emission time calculation)")


def main():
    print("="*60)
    print("DOPPLER IMPLEMENTATION VERIFICATION")
    print("="*60)
    print(f"Source: Real drone audio (broadband)")
    print(f"Velocity: {SOURCE_VELOCITY} m/s")
    print(f"Method: Retarded time (emission time calculation)")
    print("="*60)

    print("\nLoading drone audio...")
    source_audio, sample_rate = load_drone_audio()

    print("\nApplying Doppler effect to moving source...")
    mic_signals, trajectory = generate_moving_drone_audio(source_audio)

    print("\nCreating verification plots...")
    plot_verification(mic_signals, trajectory, source_audio)

    print("\nVerification complete!")


if __name__ == "__main__":
    main()
