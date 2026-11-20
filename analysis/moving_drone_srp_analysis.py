#!/usr/bin/env python3
"""
Moving Drone SRP Analysis
=========================

Analyzes SRP-based DOA tracking for a moving drone source using real drone audio.
This simulates a drone flying horizontally across a linear microphone array to
demonstrate the impact of motion on SRP beamforming accuracy.

Key differences from real_drone_srp_analysis.py:
- Source position changes over time (moving horizontally)
- Doppler shifts included in signal propagation
- Temporal analysis shows tracking performance during motion
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample
import glob
import os

# Physical constants
SPEED_OF_SOUND = 343.0  # m/s at 20°C

# Array configuration (4-mic linear array)
MIC_SPACING = 0.04  # 40mm between adjacent mics
MIC_POSITIONS = np.array([
    [-3*MIC_SPACING/2, 0],  # Mic 0: -60mm
    [-MIC_SPACING/2, 0],    # Mic 1: -20mm
    [MIC_SPACING/2, 0],     # Mic 2: +20mm
    [3*MIC_SPACING/2, 0]    # Mic 3: +60mm
])

# Moving source parameters
SOURCE_DISTANCE_Y = 50.0  # Distance perpendicular to array (m)
SOURCE_VELOCITY = 10.0     # Horizontal velocity (m/s)
SOURCE_INITIAL_X = -10.0   # Start position x-coordinate (m) - closer for better azimuth sweep
SIGNAL_DURATION = 2.0      # Duration of audio to use (seconds)

# Processing parameters
TARGET_SAMPLE_RATE = 48000
AZIMUTH_RESOLUTION = 2  # degrees (coarser for speed)
WINDOW_DURATION = 0.1   # seconds (short to capture motion)
HOP_DURATION = 0.02     # seconds (50% overlap)


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
    if sample_rate != TARGET_SAMPLE_RATE:
        num_samples = int(len(audio) * TARGET_SAMPLE_RATE / sample_rate)
        audio = resample(audio, num_samples)
        print(f"Resampled from {sample_rate} Hz to {TARGET_SAMPLE_RATE} Hz")
        sample_rate = TARGET_SAMPLE_RATE

    # Limit duration
    max_samples = int(SIGNAL_DURATION * sample_rate)
    audio = audio[:max_samples]

    print(f"Audio duration: {len(audio)/sample_rate:.2f} s")
    print(f"Audio amplitude range: [{np.min(audio):.3f}, {np.max(audio):.3f}]")

    return audio, sample_rate


def compute_source_trajectory(num_samples, sample_rate):
    """Compute source position over time"""
    t = np.arange(num_samples) / sample_rate
    x = SOURCE_INITIAL_X + SOURCE_VELOCITY * t
    y = np.ones_like(t) * SOURCE_DISTANCE_Y

    # Compute distance from array center and azimuth
    distance = np.sqrt(x**2 + y**2)
    azimuth = np.degrees(np.arctan2(x, y))  # North=0°, East=90°
    azimuth = azimuth % 360  # Wrap to [0, 360)

    return t, x, y, distance, azimuth


def simulate_moving_source_signals(audio, sample_rate):
    """
    Simulate microphone array signals for a moving source with Doppler effect.

    Uses proper acoustic Doppler: f_observed = f_source × c / (c - v_radial)
    Implemented via retarded time (emission time) calculation.

    Returns:
        mic_signals: [4, num_samples] array
        trajectory: dict with t, x, y, distance, azimuth
    """
    num_samples = len(audio)
    t_receiver = np.arange(num_samples) / sample_rate  # Receiver time

    # Source trajectory (emission times)
    t_source = t_receiver  # Initialize
    x_source = SOURCE_INITIAL_X + SOURCE_VELOCITY * t_source
    y_source = np.ones_like(t_source) * SOURCE_DISTANCE_Y
    distance = np.sqrt(x_source**2 + y_source**2)
    azimuth = np.degrees(np.arctan2(x_source, y_source)) % 360

    mic_signals = np.zeros((4, num_samples))

    for mic_idx in range(4):
        mic_pos = MIC_POSITIONS[mic_idx]

        # For each receiver time, compute emission time using retarded time
        # Receiver observes signal at time t_r that was emitted at time t_e
        # Relationship: t_r = t_e + r(t_e)/c
        # This is solved iteratively

        t_emission = np.zeros(num_samples)

        for i in range(num_samples):
            t_r = t_receiver[i]

            # Initial guess: assume source at current position
            t_e = t_r

            # Iterate to find emission time (usually converges in 2-3 iterations)
            for iteration in range(5):
                # Source position at emission time
                x_e = SOURCE_INITIAL_X + SOURCE_VELOCITY * t_e
                y_e = SOURCE_DISTANCE_Y

                # Distance from source (at emission) to mic
                dx = x_e - mic_pos[0]
                dy = y_e - mic_pos[1]
                r_e = np.sqrt(dx**2 + dy**2)

                # Propagation time
                t_prop = r_e / SPEED_OF_SOUND

                # Update emission time
                t_e_new = t_r - t_prop

                # Check convergence
                if abs(t_e_new - t_e) < 1e-9:
                    break

                t_e = t_e_new

            t_emission[i] = t_e

        # Now sample the source signal at emission times
        # This naturally includes Doppler effect:
        # - When source approaches: emission times are compressed → higher frequency
        # - When source recedes: emission times are stretched → lower frequency

        # Convert emission times to sample indices
        emission_samples = t_emission * sample_rate

        # Clip to valid range
        emission_samples = np.clip(emission_samples, 0, num_samples - 1)

        # Interpolate source signal at emission times
        doppler_signal = np.interp(emission_samples, np.arange(num_samples), audio)

        # Compute distance at emission time for attenuation
        x_e = SOURCE_INITIAL_X + SOURCE_VELOCITY * t_emission
        y_e = SOURCE_DISTANCE_Y
        dx = x_e - mic_pos[0]
        dy = y_e - mic_pos[1]
        r_e = np.sqrt(dx**2 + dy**2)

        # Apply 1/r attenuation
        attenuation = 1.0 / r_e

        # Final signal at microphone
        mic_signals[mic_idx, :] = doppler_signal * attenuation

    trajectory = {
        't': t_receiver,
        'x': x_source,
        'y': y_source,
        'distance': distance,
        'azimuth': azimuth
    }

    return mic_signals, trajectory


def compute_srp_for_window(window_signals, sample_rate, azimuth_resolution=2):
    """
    Compute SRP beamforming for a single time window.
    Returns azimuth array and SRP power spectrum.
    """
    azimuths = np.arange(0, 360, azimuth_resolution)
    srp_spectrum = np.zeros(len(azimuths))

    num_samples = window_signals.shape[1]

    for az_idx, azimuth in enumerate(azimuths):
        # Convert to radians (0° = North, 90° = East)
        theta_rad = np.radians(90 - azimuth)
        source_direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

        # Compute steering delays for each microphone
        delays = np.zeros(4)
        for mic_idx in range(4):
            # Project mic position onto source direction
            projection = np.dot(MIC_POSITIONS[mic_idx], source_direction)
            delays[mic_idx] = -projection / SPEED_OF_SOUND

        # Reference to first mic
        delays = delays - delays[0]

        # Apply delays and sum (delay-and-sum beamforming)
        beamformed = np.zeros(num_samples)
        for mic_idx in range(4):
            delay_samples = int(delays[mic_idx] * sample_rate)
            if delay_samples >= 0:
                beamformed[delay_samples:] += window_signals[mic_idx, :-delay_samples if delay_samples > 0 else None]
            else:
                beamformed[:delay_samples] += window_signals[mic_idx, -delay_samples:]

        # Compute power
        srp_spectrum[az_idx] = np.mean(beamformed**2)

    # Normalize
    srp_spectrum = srp_spectrum / np.max(srp_spectrum)

    # Find peak
    peak_idx = np.argmax(srp_spectrum)
    peak_azimuth = azimuths[peak_idx]

    return azimuths, srp_spectrum, peak_azimuth


def compute_srp_over_time(mic_signals, sample_rate, window_duration=0.1,
                         hop_duration=0.05, azimuth_resolution=2):
    """
    Compute SRP over sliding time windows for moving source tracking.
    """
    window_samples = int(window_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate)
    num_samples = mic_signals.shape[1]

    num_windows = (num_samples - window_samples) // hop_samples + 1

    azimuths = np.arange(0, 360, azimuth_resolution)
    srp_matrix = np.zeros((num_windows, len(azimuths)))
    peak_azimuths = np.zeros(num_windows)
    times = np.zeros(num_windows)

    print(f"\nComputing SRP over time:")
    print(f"  Window duration: {window_duration*1000:.1f} ms")
    print(f"  Hop duration: {hop_duration*1000:.1f} ms")
    print(f"  Number of windows: {num_windows}")

    for win_idx in range(num_windows):
        start_sample = win_idx * hop_samples
        end_sample = start_sample + window_samples
        times[win_idx] = (start_sample + window_samples/2) / sample_rate

        window_signals = mic_signals[:, start_sample:end_sample]

        _, srp_spectrum, peak_azimuth = compute_srp_for_window(
            window_signals, sample_rate, azimuth_resolution
        )

        srp_matrix[win_idx, :] = srp_spectrum
        peak_azimuths[win_idx] = peak_azimuth

        if (win_idx + 1) % 10 == 0 or win_idx == num_windows - 1:
            print(f"  Processed {win_idx+1}/{num_windows} windows")

    return times, azimuths, srp_matrix, peak_azimuths


def plot_moving_source_analysis(mic_signals, trajectory, times, azimuths,
                                srp_matrix, peak_azimuths, sample_rate):
    """
    Create comprehensive visualization of moving source SRP analysis.
    """
    fig = plt.figure(figsize=(16, 12))

    # Get true azimuth at window center times
    true_azimuths = np.interp(times, trajectory['t'], trajectory['azimuth'])

    # Handle 180° ambiguity for linear array
    peak_azimuths_alt = (peak_azimuths + 180) % 360

    # Compute errors for both ambiguous solutions
    errors1 = peak_azimuths - true_azimuths
    errors2 = peak_azimuths_alt - true_azimuths

    # Wrap errors to [-180, 180]
    errors1 = ((errors1 + 180) % 360) - 180
    errors2 = ((errors2 + 180) % 360) - 180

    # Choose best match at each time
    best_errors = np.where(np.abs(errors1) < np.abs(errors2), errors1, errors2)
    best_estimates = np.where(np.abs(errors1) < np.abs(errors2), peak_azimuths, peak_azimuths_alt)

    # === Panel 1: Trajectory geometry ===
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(trajectory['x'], trajectory['y'], 'b-', linewidth=2, label='Source trajectory')
    ax1.plot(trajectory['x'][0], trajectory['y'][0], 'go', markersize=12, label='Start')
    ax1.plot(trajectory['x'][-1], trajectory['y'][-1], 'ro', markersize=12, label='End')
    ax1.plot(MIC_POSITIONS[:, 0], MIC_POSITIONS[:, 1], 'ks', markersize=10, label='Mics')
    ax1.plot(0, 0, 'k^', markersize=12, label='Array center')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Source Trajectory')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')

    # === Panel 2: True azimuth over time ===
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(trajectory['t'], trajectory['azimuth'], 'g-', linewidth=2, label='True azimuth')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Azimuth (degrees)')
    ax2.set_title('True Azimuth vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 360])

    # === Panel 3: Distance and velocity ===
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(trajectory['t'], trajectory['distance'], 'b-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Distance (m)', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.set_title('Distance from Array Center')
    ax3.grid(True, alpha=0.3)

    # Add angular velocity on secondary axis
    angular_velocity = np.gradient(trajectory['azimuth'], trajectory['t'])
    ax3b = ax3.twinx()
    ax3b.plot(trajectory['t'], angular_velocity, 'r-', linewidth=1, alpha=0.7)
    ax3b.set_ylabel('Angular Velocity (deg/s)', color='r')
    ax3b.tick_params(axis='y', labelcolor='r')

    # === Panel 4: SRP heatmap over time ===
    ax4 = plt.subplot(3, 3, 4)
    im = ax4.imshow(srp_matrix.T, aspect='auto', origin='lower', cmap='hot',
                    extent=[times[0], times[-1], azimuths[0], azimuths[-1]],
                    interpolation='bilinear')

    # Overlay true azimuth
    ax4.plot(trajectory['t'], trajectory['azimuth'], 'c--', linewidth=2, label='True azimuth')

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Azimuth (degrees)')
    ax4.set_title('SRP Power Over Time')
    plt.colorbar(im, ax=ax4, label='Normalized Power')
    ax4.legend(loc='upper right')

    # === Panel 5: Tracked DOA (both ambiguous peaks) ===
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(trajectory['t'], trajectory['azimuth'], 'g-', linewidth=2, label='True azimuth')
    ax5.plot(times, peak_azimuths, 'ro-', markersize=4, label='SRP Peak 1')
    ax5.plot(times, peak_azimuths_alt, 'rs--', markersize=4, alpha=0.5, label='SRP Peak 2 (±180°)')
    ax5.plot(times, best_estimates, 'b*', markersize=8, label='Best match')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Azimuth (degrees)')
    ax5.set_title('DOA Tracking (with 180° ambiguity)')
    ax5.set_ylim([0, 360])
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # === Panel 6: Tracking error ===
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(times, best_errors, 'b-', linewidth=2)
    ax6.axhline(0, color='k', linestyle='--', linewidth=1)
    ax6.fill_between(times, best_errors, 0, where=(best_errors >= 0), alpha=0.3, color='red', label='Positive error')
    ax6.fill_between(times, best_errors, 0, where=(best_errors < 0), alpha=0.3, color='blue', label='Negative error')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Error (degrees)')
    ax6.set_title('DOA Tracking Error (best match)')
    ax6.grid(True, alpha=0.3)

    # Add statistics
    mean_error = np.mean(best_errors)
    std_error = np.std(best_errors)
    rms_error = np.sqrt(np.mean(best_errors**2))
    max_abs_error = np.max(np.abs(best_errors))

    stats_text = f'Mean: {mean_error:+.1f}°\nStd: {std_error:.1f}°\nRMS: {rms_error:.1f}°\nMax: {max_abs_error:.1f}°'
    ax6.text(0.98, 0.02, stats_text, transform=ax6.transAxes,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9, family='monospace')

    # === Panel 7: Microphone signals (first 0.1s) ===
    ax7 = plt.subplot(3, 3, 7)
    t_plot = np.arange(int(0.1 * sample_rate)) / sample_rate
    for mic_idx in range(4):
        ax7.plot(t_plot, mic_signals[mic_idx, :len(t_plot)] + mic_idx*0.5,
                label=f'Mic {mic_idx}')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Amplitude (offset)')
    ax7.set_title('Microphone Signals (first 100ms)')
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    # === Panel 8: Error vs distance ===
    ax8 = plt.subplot(3, 3, 8)
    distances_at_times = np.interp(times, trajectory['t'], trajectory['distance'])
    ax8.scatter(distances_at_times, np.abs(best_errors), c=times, cmap='viridis', s=50)
    ax8.set_xlabel('Distance from Array (m)')
    ax8.set_ylabel('Absolute Error (degrees)')
    ax8.set_title('Error vs Distance')
    ax8.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax8.collections[0], ax=ax8, label='Time (s)')

    # === Panel 9: Error vs angular velocity ===
    ax9 = plt.subplot(3, 3, 9)
    angular_vel_at_times = np.interp(times, trajectory['t'], angular_velocity)
    ax9.scatter(angular_vel_at_times, np.abs(best_errors), c=times, cmap='viridis', s=50)
    ax9.set_xlabel('Angular Velocity (deg/s)')
    ax9.set_ylabel('Absolute Error (degrees)')
    ax9.set_title('Error vs Angular Velocity')
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('moving_drone_srp_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: moving_drone_srp_analysis.png")

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"MOVING SOURCE SRP TRACKING SUMMARY")
    print(f"{'='*60}")
    print(f"Signal duration: {trajectory['t'][-1]:.2f} s")
    print(f"Source velocity: {SOURCE_VELOCITY} m/s")
    print(f"Distance range: {np.min(trajectory['distance']):.1f} - {np.max(trajectory['distance']):.1f} m")
    print(f"Azimuth range: {np.min(trajectory['azimuth']):.1f}° - {np.max(trajectory['azimuth']):.1f}°")
    print(f"Angular velocity range: {np.min(angular_velocity):.1f} - {np.max(angular_velocity):.1f} deg/s")
    print(f"\nTracking Performance (with 180° ambiguity resolution):")
    print(f"  Mean error: {mean_error:+.1f}°")
    print(f"  Std error: {std_error:.1f}°")
    print(f"  RMS error: {rms_error:.1f}°")
    print(f"  Max absolute error: {max_abs_error:.1f}°")
    print(f"{'='*60}")


def main():
    print("="*60)
    print("MOVING DRONE SRP ANALYSIS")
    print("="*60)
    print(f"Array configuration: 4-mic linear array, {MIC_SPACING*1000:.0f}mm spacing")
    print(f"Source velocity: {SOURCE_VELOCITY} m/s")
    print(f"Perpendicular distance: {SOURCE_DISTANCE_Y} m")
    print(f"Initial x-position: {SOURCE_INITIAL_X} m")
    print(f"Signal duration: {SIGNAL_DURATION} s")
    print("="*60)

    # Load drone audio
    audio, sample_rate = load_drone_audio()

    # Simulate moving source
    print("\nSimulating moving source with Doppler effect...")
    mic_signals, trajectory = simulate_moving_source_signals(audio, sample_rate)
    print(f"Trajectory: {trajectory['x'][0]:.1f}m to {trajectory['x'][-1]:.1f}m")
    print(f"Azimuth sweep: {trajectory['azimuth'][0]:.1f}° to {trajectory['azimuth'][-1]:.1f}°")

    # Compute SRP over time
    times, azimuths, srp_matrix, peak_azimuths = compute_srp_over_time(
        mic_signals, sample_rate,
        window_duration=WINDOW_DURATION,
        hop_duration=HOP_DURATION,
        azimuth_resolution=AZIMUTH_RESOLUTION
    )

    # Plot comprehensive analysis
    print("\nGenerating comprehensive analysis plots...")
    plot_moving_source_analysis(mic_signals, trajectory, times, azimuths,
                                srp_matrix, peak_azimuths, sample_rate)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
