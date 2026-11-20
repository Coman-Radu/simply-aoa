"""
Experiment 3: Single Moving Source

Single drone audio source starting at 30deg, 10m, moving at 5 m/s tangential
velocity. Uses real drone WAV recording as the source signal.
Processes signal in 0.2s windows and tracks azimuth over time.
Generates azimuth-vs-time + polar trajectory (1x2 figure).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from sim_common import (
    load_source_wav, get_mic_positions_4x4, beamforming_scan, detect_peaks,
    SOURCE_AZIMUTH_INITIAL, SOURCE_DISTANCE, SOURCE_VELOCITY, SNR_DB,
    MIC_SPACING, SIMULATION_DURATION, TIME_WINDOW, SAMPLE_RATE, CHANNELS,
    SPEED_OF_SOUND
)


def simulate_moving_source(duration, initial_azimuth_deg, distance, velocity):
    """
    Simulate a source moving at constant tangential velocity around the array.

    Uses the real drone WAV recording as source signal.
    The source stays at constant distance and sweeps in azimuth.
    Angular velocity = velocity / distance (rad/s).

    Returns:
        time_axis: Time vector
        mic_signals: [samples, channels]
        true_azimuths: True azimuth at each sample
    """
    mic_positions = get_mic_positions_4x4()
    num_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Angular velocity in deg/s
    angular_velocity = np.degrees(velocity / distance)
    true_azimuths = initial_azimuth_deg + angular_velocity * t

    # Load drone WAV as source signal (looped to match duration)
    _, source_signal = load_source_wav(duration=duration)

    signal_power = np.mean(source_signal ** 2)
    noise_power = signal_power / (10 ** (SNR_DB / 10))
    noise_std = np.sqrt(noise_power)

    mic_signals = np.zeros((num_samples, CHANNELS))

    for ch in range(CHANNELS):
        pos = np.array(mic_positions[ch])
        ch_signal = np.zeros(num_samples)

        for i in range(num_samples):
            az_rad = np.radians(90 - true_azimuths[i])
            source_dir = np.array([np.cos(az_rad), np.sin(az_rad)])
            distance_proj = np.dot(pos, source_dir)

            # Fractional delay via linear interpolation
            time_delay = distance_proj / SPEED_OF_SOUND
            delay_samples_frac = time_delay * SAMPLE_RATE
            src_idx = i + delay_samples_frac

            idx_low = int(np.floor(src_idx))
            frac = src_idx - idx_low

            if 0 <= idx_low < num_samples - 1:
                ch_signal[i] = ((1 - frac) * source_signal[idx_low] +
                                frac * source_signal[idx_low + 1])
            elif idx_low == num_samples - 1:
                ch_signal[i] = source_signal[idx_low]

            mic_distance = distance + distance_proj
            ch_signal[i] *= distance / mic_distance

        noise = np.random.randn(num_samples) * noise_std
        mic_signals[:, ch] = ch_signal + noise

    return t, mic_signals, true_azimuths


def main():
    np.random.seed(42)

    angular_vel = np.degrees(SOURCE_VELOCITY / SOURCE_DISTANCE)

    print("=" * 60)
    print("EXPERIMENT 3: Single Moving Source")
    print(f"  Initial azimuth = {SOURCE_AZIMUTH_INITIAL}deg, "
          f"Distance = {SOURCE_DISTANCE}m")
    print(f"  Tangential velocity = {SOURCE_VELOCITY} m/s "
          f"(angular vel = {angular_vel:.1f} deg/s)")
    print(f"  Duration = {SIMULATION_DURATION}s, Window = {TIME_WINDOW}s")
    print(f"  SNR = {SNR_DB}dB, Mic spacing = {MIC_SPACING*1000:.0f}mm")
    print("=" * 60)

    t, mic_signals, true_azimuths = simulate_moving_source(
        SIMULATION_DURATION, SOURCE_AZIMUTH_INITIAL, SOURCE_DISTANCE,
        SOURCE_VELOCITY)

    # Process in time windows
    window_samples = int(TIME_WINDOW * SAMPLE_RATE)
    num_windows = len(t) // window_samples
    time_centers = []
    estimated_azimuths = []
    true_az_centers = []

    for w in range(num_windows):
        start = w * window_samples
        end = start + window_samples
        segment = mic_signals[start:end, :]

        az_scan, power = beamforming_scan(segment)
        peaks, _ = detect_peaks(az_scan, power)

        t_center = (t[start] + t[end - 1]) / 2
        true_az_center = true_azimuths[start + window_samples // 2]
        time_centers.append(t_center)
        true_az_centers.append(true_az_center)

        if len(peaks) > 0:
            estimated_azimuths.append(peaks[0]['azimuth'])
        else:
            estimated_azimuths.append(np.nan)

    time_centers = np.array(time_centers)
    estimated_azimuths = np.array(estimated_azimuths)
    true_az_centers = np.array(true_az_centers)

    # RMS error (ignoring NaN)
    valid = ~np.isnan(estimated_azimuths)
    if np.any(valid):
        errors = np.abs(estimated_azimuths[valid] - true_az_centers[valid])
        errors = np.where(errors > 180, 360 - errors, errors)
        rms_error = np.sqrt(np.mean(errors ** 2))
        print(f"\n  RMS tracking error: {rms_error:.1f}deg")
        print(f"  Valid detections: {np.sum(valid)}/{num_windows}")
    else:
        rms_error = np.nan
        print("\n  No valid detections!")

    # ── Plot: 1x2 figure ───────────────────────────────────────────────
    fig, (ax3, _) = plt.subplots(1, 2, figsize=(14, 5))
    _.remove()
    ax3p = fig.add_subplot(1, 2, 2, projection='polar')

    # Azimuth vs time
    ax3.plot(t, true_azimuths, 'g-', linewidth=2, label='True azimuth')
    valid_mask = ~np.isnan(estimated_azimuths)
    ax3.plot(time_centers[valid_mask], estimated_azimuths[valid_mask], 'ro',
             markersize=6, label=f'Estimated (RMS err = {rms_error:.1f}deg)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Azimuth (degrees)')
    ax3.set_title('Moving Source - Azimuth vs Time',
                  fontsize=12, weight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Polar trajectory
    ax3p.set_theta_zero_location('N')
    ax3p.set_theta_direction(-1)

    # Time-coded trajectory
    cmap = plt.cm.viridis
    norm = plt.Normalize(t[0], t[-1])
    for i in range(len(t) - 1):
        ax3p.plot([np.radians(true_azimuths[i]),
                   np.radians(true_azimuths[i + 1])],
                  [SOURCE_DISTANCE, SOURCE_DISTANCE],
                  color=cmap(norm(t[i])), linewidth=2)

    ax3p.plot(np.radians(true_azimuths[0]), SOURCE_DISTANCE, 'g^',
              markersize=12, label='Start')
    ax3p.plot(np.radians(true_azimuths[-1]), SOURCE_DISTANCE, 'rs',
              markersize=12, label='End')

    if np.any(valid_mask):
        ax3p.scatter(np.radians(estimated_azimuths[valid_mask]),
                     np.full(np.sum(valid_mask), SOURCE_DISTANCE * 0.9),
                     c=time_centers[valid_mask], cmap='viridis', s=40,
                     zorder=5, edgecolors='red', linewidths=0.5)

    ax3p.set_title('Polar Trajectory', fontsize=12, weight='bold', pad=20)
    ax3p.legend(loc='lower right', fontsize=8)

    fig.suptitle(
        f'Experiment 3: Moving Source (Drone Audio)\n'
        f'4x4 Array, {SOURCE_DISTANCE}m, v={SOURCE_VELOCITY}m/s, SNR={SNR_DB}dB',
        fontsize=13, weight='bold', y=1.02)

    plt.tight_layout()
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'exp3_moving_source.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  Saved plot to: {output_file}")
    plt.show()


if __name__ == "__main__":
    main()
