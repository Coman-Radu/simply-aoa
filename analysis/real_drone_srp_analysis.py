"""
Real Drone Audio SRP Analysis

Uses actual drone audio file with simulated array geometry to demonstrate
Steered Response Power (SRP) beamforming for DOA estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate, resample
import glob

# Constants
SPEED_OF_SOUND = 343.0  # m/s
MIC_SPACING = 0.04  # meters (40mm between mics)
SAMPLE_RATE_TARGET = 48000  # Target sample rate

# Array configuration - 4 mics in a line along x-axis
mic_positions = np.array([
    [-1.5 * MIC_SPACING, 0],  # Mic 0
    [-0.5 * MIC_SPACING, 0],  # Mic 1
    [0.5 * MIC_SPACING, 0],   # Mic 2
    [1.5 * MIC_SPACING, 0]    # Mic 3
])

NUM_MICS = 4

print("=" * 80)
print("REAL DRONE AUDIO - STEERED RESPONSE POWER ANALYSIS")
print("=" * 80)
print(f"Array configuration: {NUM_MICS} microphones in line")
print(f"Microphone spacing: {MIC_SPACING*1000:.1f} mm")
print(f"Total array aperture: {3*MIC_SPACING*1000:.1f} mm")
print("=" * 80)
print()


def load_drone_audio():
    """
    Load the drone audio file from the folder.

    Returns:
        sample_rate: Sample rate of the audio
        audio: Audio data (mono)
    """
    # Find the drone audio file
    wav_files = glob.glob("Binary*.wav") + glob.glob("binary*.wav")

    if not wav_files:
        wav_files = glob.glob("*.wav")

    if not wav_files:
        raise FileNotFoundError("No WAV files found in directory")

    wav_file = wav_files[0]
    print(f"Loading audio file: {wav_file}")

    sample_rate, audio = wavfile.read(wav_file)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Normalize
    audio = audio.astype(float)
    audio = audio / (np.max(np.abs(audio)) + 1e-10)

    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {len(audio)/sample_rate:.2f} seconds")
    print(f"  Samples: {len(audio)}")
    print()

    return sample_rate, audio, wav_file


def simulate_array_signals(audio, sample_rate, source_azimuth, source_distance=10.0,
                           signal_duration=0.5, start_time=0.5):
    """
    Simulate what each microphone receives from a stationary source.

    Args:
        audio: Source audio signal
        sample_rate: Sample rate
        source_azimuth: True source azimuth in degrees (0° = North/+y, 90° = East/+x)
        source_distance: Distance to source in meters
        signal_duration: Duration of signal to use in seconds
        start_time: Start time in audio file

    Returns:
        mic_signals: Array of signals [num_mics, num_samples]
        source_position: [x, y] position of source
    """
    # Calculate source position
    theta_rad = np.radians(90 - source_azimuth)  # Convert to standard coordinates
    source_position = np.array([
        source_distance * np.cos(theta_rad),
        source_distance * np.sin(theta_rad)
    ])

    print(f"Source configuration:")
    print(f"  Azimuth: {source_azimuth:.1f}°")
    print(f"  Distance: {source_distance:.1f} m")
    print(f"  Position: ({source_position[0]:.2f}, {source_position[1]:.2f}) m")
    print()

    # Extract segment of audio
    start_sample = int(start_time * sample_rate)
    num_samples = int(signal_duration * sample_rate)

    if start_sample + num_samples > len(audio):
        num_samples = len(audio) - start_sample

    audio_segment = audio[start_sample:start_sample + num_samples]

    # Simulate each microphone
    mic_signals = []

    for mic_idx in range(NUM_MICS):
        mic_pos = mic_positions[mic_idx]

        # Calculate distance from source to this mic
        distance = np.linalg.norm(source_position - mic_pos)

        # Time delay (propagation time)
        time_delay = distance / SPEED_OF_SOUND
        delay_samples = int(time_delay * sample_rate)

        # Apply delay
        if delay_samples > 0:
            delayed = np.pad(audio_segment, (delay_samples, 0), mode='constant')[:num_samples]
        else:
            delayed = audio_segment

        # Amplitude attenuation (1/r)
        amplitude = source_distance / distance
        delayed = delayed * amplitude

        # Add slight noise
        noise_level = 0.02 * np.std(delayed)
        delayed = delayed + noise_level * np.random.randn(len(delayed))

        mic_signals.append(delayed)

    mic_signals = np.array(mic_signals)

    print(f"Simulated {NUM_MICS} microphone signals")
    print(f"  Signal duration: {signal_duration} s")
    print(f"  Samples per mic: {num_samples}")
    print()

    return mic_signals, source_position


def compute_srp_beamforming(mic_signals, sample_rate, azimuth_resolution=1):
    """
    Compute Steered Response Power across all azimuth angles.

    Args:
        mic_signals: Array of microphone signals [num_mics, num_samples]
        sample_rate: Sample rate
        azimuth_resolution: Angular resolution in degrees

    Returns:
        azimuths: Array of azimuth angles
        srp_power: SRP power at each azimuth
        peak_azimuth: Azimuth with maximum power
        peak_power: Maximum power value
    """
    print("Computing Steered Response Power...")

    azimuths = np.arange(0, 360, azimuth_resolution)
    srp_power = np.zeros(len(azimuths))

    num_samples = mic_signals.shape[1]

    for az_idx, azimuth in enumerate(azimuths):
        # Convert azimuth to direction vector
        theta_rad = np.radians(90 - azimuth)
        source_direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

        # Apply steering delays
        delayed_signals = []

        for mic_idx in range(NUM_MICS):
            mic_pos = mic_positions[mic_idx]

            # Calculate required delay for this direction
            distance_proj = np.dot(mic_pos, source_direction)
            time_delay = -distance_proj / SPEED_OF_SOUND
            delay_samples = int(time_delay * sample_rate)

            # Apply delay
            if delay_samples > 0:
                delayed = np.pad(mic_signals[mic_idx], (delay_samples, 0), mode='constant')[:num_samples]
            elif delay_samples < 0:
                delayed = np.pad(mic_signals[mic_idx], (0, -delay_samples), mode='constant')[-num_samples:]
            else:
                delayed = mic_signals[mic_idx]

            delayed_signals.append(delayed)

        # Delay-and-sum beamforming
        beamformed = np.mean(delayed_signals, axis=0)

        # Calculate power
        srp_power[az_idx] = np.mean(beamformed**2)

        if (az_idx + 1) % 45 == 0:
            print(f"  Progress: {az_idx + 1}/{len(azimuths)} azimuths")

    # Normalize
    srp_power = srp_power / (np.max(srp_power) + 1e-10)

    # Find peak
    peak_idx = np.argmax(srp_power)
    peak_azimuth = azimuths[peak_idx]
    peak_power = srp_power[peak_idx]

    # For linear array: 180° ambiguity - check both peaks
    peak_azimuth_alt = (peak_azimuth + 180) % 360

    print(f"\nSRP Results:")
    print(f"  Primary peak: {peak_azimuth:.1f}° (power: {peak_power:.3f})")
    print(f"  180° ambiguity peak: {peak_azimuth_alt:.1f}°")
    print(f"  Note: Linear array cannot distinguish these two directions")
    print()

    return azimuths, srp_power, peak_azimuth, peak_power


def compute_srp_over_time(mic_signals, sample_rate, window_duration=0.1,
                         hop_duration=0.05, azimuth_resolution=2):
    """
    Compute SRP over sliding time windows to show evolution.

    Args:
        mic_signals: Array of microphone signals [num_mics, num_samples]
        sample_rate: Sample rate
        window_duration: Duration of each analysis window in seconds
        hop_duration: Time between consecutive windows in seconds
        azimuth_resolution: Angular resolution in degrees

    Returns:
        times: Center time of each window
        azimuths: Azimuth angles
        srp_matrix: 2D array [num_windows, num_azimuths] of SRP power
        peak_azimuths: Peak azimuth for each time window
    """
    print(f"\nComputing SRP over time...")
    print(f"  Window duration: {window_duration} s")
    print(f"  Hop duration: {hop_duration} s")
    print(f"  Azimuth resolution: {azimuth_resolution}°")

    window_samples = int(window_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate)

    num_samples = mic_signals.shape[1]
    num_windows = (num_samples - window_samples) // hop_samples + 1

    azimuths = np.arange(0, 360, azimuth_resolution)
    srp_matrix = np.zeros((num_windows, len(azimuths)))
    peak_azimuths = np.zeros(num_windows)
    times = np.zeros(num_windows)

    for win_idx in range(num_windows):
        start_sample = win_idx * hop_samples
        end_sample = start_sample + window_samples

        if end_sample > num_samples:
            break

        # Extract window
        window_signals = mic_signals[:, start_sample:end_sample]

        # Compute SRP for this window
        for az_idx, azimuth in enumerate(azimuths):
            theta_rad = np.radians(90 - azimuth)
            source_direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

            delayed_signals = []
            for mic_idx in range(NUM_MICS):
                mic_pos = mic_positions[mic_idx]
                distance_proj = np.dot(mic_pos, source_direction)
                time_delay = -distance_proj / SPEED_OF_SOUND
                delay_samples = int(time_delay * sample_rate)

                if delay_samples > 0:
                    delayed = np.pad(window_signals[mic_idx], (delay_samples, 0),
                                   mode='constant')[:window_samples]
                elif delay_samples < 0:
                    delayed = np.pad(window_signals[mic_idx], (0, -delay_samples),
                                   mode='constant')[-window_samples:]
                else:
                    delayed = window_signals[mic_idx]

                delayed_signals.append(delayed)

            beamformed = np.mean(delayed_signals, axis=0)
            srp_matrix[win_idx, az_idx] = np.mean(beamformed**2)

        # Normalize this window
        srp_matrix[win_idx, :] /= (np.max(srp_matrix[win_idx, :]) + 1e-10)

        # Find peak for this window
        peak_idx = np.argmax(srp_matrix[win_idx, :])
        peak_azimuths[win_idx] = azimuths[peak_idx]

        # Time at center of window
        times[win_idx] = (start_sample + window_samples/2) / sample_rate

        if (win_idx + 1) % 5 == 0:
            print(f"  Progress: {win_idx + 1}/{num_windows} windows")

    print(f"  Completed {num_windows} time windows")
    print()

    return times, azimuths, srp_matrix, peak_azimuths


def plot_results(audio_segment, mic_signals, sample_rate, azimuths, srp_power,
                true_azimuth, estimated_azimuth, source_position, wav_filename,
                best_estimate, best_error):
    """
    Plot comprehensive analysis results.
    """
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Source audio waveform
    ax1 = fig.add_subplot(4, 2, 1)
    time_audio = np.arange(len(audio_segment)) / sample_rate * 1000  # ms
    ax1.plot(time_audio, audio_segment, 'b-', linewidth=0.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Source Audio Waveform (Drone)', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Source audio spectrum
    ax2 = fig.add_subplot(4, 2, 2)
    from scipy.fft import fft, fftfreq
    spectrum = np.abs(fft(audio_segment))
    freqs = fftfreq(len(audio_segment), 1/sample_rate)

    # Only positive frequencies up to 8 kHz
    mask = (freqs > 0) & (freqs < 8000)
    ax2.plot(freqs[mask], 20*np.log10(spectrum[mask] + 1e-10), 'b-', linewidth=1)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('Source Audio Spectrum', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3-6: Individual microphone signals (first 50ms)
    for mic_idx in range(NUM_MICS):
        ax = fig.add_subplot(4, 2, 3 + mic_idx)

        # Show first 50ms
        n_samples_show = int(0.05 * sample_rate)
        time_ms = np.arange(n_samples_show) / sample_rate * 1000

        ax.plot(time_ms, mic_signals[mic_idx, :n_samples_show], linewidth=0.8)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Mic {mic_idx} @ x={mic_positions[mic_idx][0]*1000:.0f}mm',
                    fontsize=11, weight='bold')
        ax.grid(True, alpha=0.3)

    # Plot 7: Array geometry
    ax7 = fig.add_subplot(4, 2, 7)
    ax7.plot(mic_positions[:, 0], mic_positions[:, 1], 'ko', markersize=12)
    for i in range(NUM_MICS):
        ax7.text(mic_positions[i, 0], mic_positions[i, 1] - 0.005,
                f'M{i}', ha='center', va='top', fontsize=9)

    # Plot source
    ax7.plot(source_position[0], source_position[1], 'r*', markersize=20,
            label=f'Source at {true_azimuth:.0f}°')

    # Draw line from array center to source
    ax7.plot([0, source_position[0]], [0, source_position[1]],
            'r--', linewidth=2, alpha=0.5)

    ax7.set_xlabel('X (m)')
    ax7.set_ylabel('Y (m)')
    ax7.set_title('Array Geometry', fontsize=12, weight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.axis('equal')

    # Plot 8: SRP Envelope (main result)
    ax8 = fig.add_subplot(4, 2, 8)
    ax8.plot(azimuths, srp_power, 'b-', linewidth=3, label='SRP Power')

    # Mark true DOA
    ax8.axvline(true_azimuth, color='green', linestyle='--', linewidth=3,
                label=f'True DOA: {true_azimuth:.0f}°', alpha=0.8, zorder=10)

    # Calculate 180° ambiguity
    estimated_azimuth_alt = (estimated_azimuth + 180) % 360

    # Calculate errors
    error1 = estimated_azimuth - true_azimuth
    if error1 > 180:
        error1 -= 360
    elif error1 < -180:
        error1 += 360

    error2 = estimated_azimuth_alt - true_azimuth
    if error2 > 180:
        error2 -= 360
    elif error2 < -180:
        error2 += 360

    # Mark both possible DOAs (180° ambiguity)
    peak_power = srp_power[np.argmin(np.abs(azimuths - estimated_azimuth))]

    # Primary peak
    ax8.plot(estimated_azimuth, peak_power, 'ro', markersize=15,
            markeredgewidth=3, markeredgecolor='darkred',
            label=f'Peak 1: {estimated_azimuth:.0f}° (Δ={error1:+.0f}°)', zorder=5)

    # Alternative peak (180° ambiguity)
    peak_power_alt = srp_power[np.argmin(np.abs(azimuths - estimated_azimuth_alt))]
    ax8.plot(estimated_azimuth_alt, peak_power_alt, 'rs', markersize=15,
            markeredgewidth=3, markeredgecolor='darkred',
            label=f'Peak 2: {estimated_azimuth_alt:.0f}° (Δ={error2:+.0f}°)', zorder=5)

    # Annotate primary peak
    ax8.annotate(f'{estimated_azimuth:.0f}°\n(error: {error1:+.0f}°)',
                xy=(estimated_azimuth, peak_power),
                xytext=(20, 20), textcoords='offset points',
                fontsize=10, color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Annotate alternative peak
    ax8.annotate(f'{estimated_azimuth_alt:.0f}°\n(error: {error2:+.0f}°)',
                xy=(estimated_azimuth_alt, peak_power_alt),
                xytext=(20, -40), textcoords='offset points',
                fontsize=10, color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Show best match
    best_error = error1 if abs(error1) < abs(error2) else error2
    best_estimate = estimated_azimuth if abs(error1) < abs(error2) else estimated_azimuth_alt

    ax8.text(0.02, 0.97, f'180° Ambiguity\nBest: {best_estimate:.0f}°\nError: {best_error:+.1f}°',
            transform=ax8.transAxes, fontsize=11, weight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax8.set_xlabel('Azimuth (degrees)', fontsize=12)
    ax8.set_ylabel('Normalized SRP Power', fontsize=12)
    ax8.set_title('Steered Response Power Envelope', fontsize=13, weight='bold')
    ax8.legend(loc='upper right', fontsize=10)
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(0, 360)
    ax8.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax8.set_ylim(-0.05, 1.1)

    plt.suptitle(f'Real Drone Audio SRP Analysis (Linear Array → 180° Ambiguity)\n'
                f'Source: {wav_filename} | True DOA: {true_azimuth:.0f}° | '
                f'Peaks: {estimated_azimuth:.0f}° & {estimated_azimuth_alt:.0f}° | Best Match Error: {best_error:+.1f}°',
                fontsize=13, weight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig('real_drone_srp_analysis.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'real_drone_srp_analysis.png'")
    plt.show()


def plot_srp_over_time(times, azimuths, srp_matrix, peak_azimuths,
                      true_azimuth, wav_filename):
    """
    Plot SRP power evolution over time as a heatmap.

    Args:
        times: Time points for each window
        azimuths: Azimuth angles
        srp_matrix: 2D array [num_windows, num_azimuths]
        peak_azimuths: Peak azimuth at each time
        true_azimuth: True source azimuth
        wav_filename: Source filename
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: SRP Heatmap (Azimuth vs Time)
    ax1 = axes[0]

    # Transpose for correct orientation: time on x-axis, azimuth on y-axis
    im = ax1.imshow(srp_matrix.T, aspect='auto', origin='lower',
                   extent=[times[0], times[-1], azimuths[0], azimuths[-1]],
                   cmap='hot', interpolation='bilinear')

    # Mark true DOA
    ax1.axhline(true_azimuth, color='cyan', linestyle='--', linewidth=2.5,
               label=f'True DOA: {true_azimuth:.0f}°', alpha=0.8)

    # Also mark 180° ambiguity
    true_azimuth_alt = (true_azimuth + 180) % 360
    ax1.axhline(true_azimuth_alt, color='lightblue', linestyle=':', linewidth=2,
               label=f'180° ambiguity: {true_azimuth_alt:.0f}°', alpha=0.8)

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Azimuth (degrees)', fontsize=12)
    ax1.set_title('SRP Power Distribution Over Time\n(Heatmap: Azimuth vs Time)',
                 fontsize=13, weight='bold')
    ax1.set_ylim(0, 360)
    ax1.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax1.legend(loc='upper right', fontsize=10)

    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Normalized SRP Power', fontsize=10)

    # Plot 2: Tracked DOA over time
    ax2 = axes[1]

    # Plot both possible DOAs (180° ambiguity)
    peak_azimuths_alt = (peak_azimuths + 180) % 360

    ax2.plot(times, peak_azimuths, 'ro-', linewidth=2, markersize=6,
            label='Peak 1', alpha=0.7)
    ax2.plot(times, peak_azimuths_alt, 'rs--', linewidth=2, markersize=6,
            label='Peak 2 (±180°)', alpha=0.7)

    # True DOA
    ax2.axhline(true_azimuth, color='green', linestyle='--', linewidth=2.5,
               label=f'True DOA: {true_azimuth:.0f}°', alpha=0.8)

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Estimated Azimuth (degrees)', fontsize=12)
    ax2.set_title('Tracked DOA Over Time (with 180° Ambiguity)',
                 fontsize=13, weight='bold')
    ax2.set_ylim(0, 360)
    ax2.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: DOA error over time (best match)
    ax3 = axes[2]

    # Calculate errors for both peaks and choose best
    errors = []
    for i in range(len(peak_azimuths)):
        error1 = peak_azimuths[i] - true_azimuth
        if error1 > 180:
            error1 -= 360
        elif error1 < -180:
            error1 += 360

        error2 = peak_azimuths_alt[i] - true_azimuth
        if error2 > 180:
            error2 -= 360
        elif error2 < -180:
            error2 += 360

        # Choose smaller error
        best_error = error1 if abs(error1) < abs(error2) else error2
        errors.append(best_error)

    errors = np.array(errors)

    ax3.plot(times, errors, 'b-', linewidth=2.5, alpha=0.7)
    ax3.fill_between(times, 0, errors, alpha=0.3)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

    # Statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    rms_error = np.sqrt(np.mean(errors**2))

    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('DOA Error (degrees)', fontsize=12)
    ax3.set_title('DOA Estimation Error Over Time (Best Match After 180° Unwrap)',
                 fontsize=13, weight='bold')
    ax3.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = (f'Mean: {mean_error:+.1f}°\n'
                 f'Std:  {std_error:.1f}°\n'
                 f'RMS:  {rms_error:.1f}°')
    ax3.text(0.02, 0.97, stats_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'Real Drone Audio: SRP Temporal Evolution (Stationary Source)\n'
                f'Source: {wav_filename} | True DOA: {true_azimuth:.0f}°',
                fontsize=13, weight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig('real_drone_srp_over_time.png', dpi=150, bbox_inches='tight')
    print("SRP over time plot saved as 'real_drone_srp_over_time.png'")
    plt.show()


def main():
    """
    Main analysis function.
    """
    # Load drone audio
    sample_rate, audio, wav_filename = load_drone_audio()

    # Resample if needed
    if sample_rate != SAMPLE_RATE_TARGET:
        print(f"Resampling from {sample_rate} Hz to {SAMPLE_RATE_TARGET} Hz...")
        num_samples_new = int(len(audio) * SAMPLE_RATE_TARGET / sample_rate)
        audio = resample(audio, num_samples_new)
        sample_rate = SAMPLE_RATE_TARGET
        print(f"  New duration: {len(audio)/sample_rate:.2f} s\n")

    # Source configuration (user can modify these)
    SOURCE_AZIMUTH = 45.0  # degrees (0° = North, 90° = East)
    SOURCE_DISTANCE = 10.0  # meters
    SIGNAL_DURATION = 0.5  # seconds
    START_TIME = 0.5  # seconds into the audio file

    print("=" * 80)
    print("SIMULATION PARAMETERS")
    print("=" * 80)
    print(f"True source azimuth: {SOURCE_AZIMUTH}°")
    print(f"Source distance: {SOURCE_DISTANCE} m")
    print(f"Signal duration: {SIGNAL_DURATION} s")
    print(f"Start time: {START_TIME} s")
    print("=" * 80)
    print()

    # Simulate array receiving the signal
    mic_signals, source_position = simulate_array_signals(
        audio, sample_rate, SOURCE_AZIMUTH, SOURCE_DISTANCE,
        SIGNAL_DURATION, START_TIME
    )

    # Get audio segment for plotting
    start_sample = int(START_TIME * sample_rate)
    num_samples = int(SIGNAL_DURATION * sample_rate)
    audio_segment = audio[start_sample:start_sample + num_samples]

    # Compute SRP (full signal)
    azimuths, srp_power, estimated_azimuth, peak_power = compute_srp_beamforming(
        mic_signals, sample_rate, azimuth_resolution=1
    )

    # Compute SRP over time
    times_srp, azimuths_time, srp_matrix, peak_azimuths_time = compute_srp_over_time(
        mic_signals, sample_rate, window_duration=0.1, hop_duration=0.02,
        azimuth_resolution=2
    )

    # Calculate error considering 180° ambiguity
    estimated_azimuth_alt = (estimated_azimuth + 180) % 360

    error1 = estimated_azimuth - SOURCE_AZIMUTH
    if error1 > 180:
        error1 -= 360
    elif error1 < -180:
        error1 += 360

    error2 = estimated_azimuth_alt - SOURCE_AZIMUTH
    if error2 > 180:
        error2 -= 360
    elif error2 < -180:
        error2 += 360

    # Choose the smaller error
    if abs(error1) < abs(error2):
        best_estimate = estimated_azimuth
        best_error = error1
    else:
        best_estimate = estimated_azimuth_alt
        best_error = error2

    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"True DOA:               {SOURCE_AZIMUTH:6.1f}°")
    print(f"\nLinear Array 180° Ambiguity:")
    print(f"  Peak 1:               {estimated_azimuth:6.1f}° (error: {error1:+6.1f}°)")
    print(f"  Peak 2 (±180°):       {estimated_azimuth_alt:6.1f}° (error: {error2:+6.1f}°)")
    print(f"\nBest Match:")
    print(f"  Estimated DOA:        {best_estimate:6.1f}°")
    print(f"  Error:                {best_error:+6.1f}°")
    print(f"  Peak power:           {peak_power:6.3f}")
    print("=" * 80)
    print()

    # Plot results
    plot_results(audio_segment, mic_signals, sample_rate, azimuths, srp_power,
                SOURCE_AZIMUTH, estimated_azimuth, source_position, wav_filename,
                best_estimate, best_error)

    # Plot SRP over time
    plot_srp_over_time(times_srp, azimuths_time, srp_matrix, peak_azimuths_time,
                      SOURCE_AZIMUTH, wav_filename)


if __name__ == "__main__":
    main()
