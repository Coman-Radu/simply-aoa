"""
SNR Filter Comparison for Beamforming

This script compares beamforming DOA estimation with and without SNR filtering.
A simulated source is placed at 100m distance, 30° azimuth from the array center.

The comparison shows how SNR filtering helps reject noise-induced false peaks
while preserving the true source direction.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks
import os

# Constants (matching rt_beamforming.py)
SPEED_OF_SOUND = 343.0  # m/s
MIC_SPACING = 0.042  # meters
SAMPLE_RATE = 48000
CHANNELS = 16

# Simulation parameters
SOURCE_DISTANCE = 100.0  # meters
SOURCE_AZIMUTH = 30.0  # degrees (0° = North, clockwise)
SNR_DB = 3  # Signal-to-noise ratio in dB for the simulation (low SNR to show filter benefit)

# Beamforming parameters
AZIMUTH_RESOLUTION = 2  # degrees
MIN_PEAK_HEIGHT = 0.1  # Lower threshold to show noise susceptibility without SNR filter
MIN_PROMINENCE = 0.05  # Lower prominence to demonstrate more noise peaks
MIN_SEPARATION = 15  # degrees

# SNR filter parameters
SNR_THRESHOLD_DB = 10.0  # Minimum SNR required for valid peak detection (higher = more selective)
NOISE_PERCENTILE = 25  # Percentile of power spectrum used as noise floor estimate


def get_mic_positions_4x4():
    """
    Generate microphone positions for 4x4 array with origin at CENTER.
    (Copied from rt_beamforming.py)
    """
    outer_spacing = MIC_SPACING
    center_offset_x = 1.5 * outer_spacing
    center_offset_y = 1.5 * outer_spacing
    inner_half_spacing = outer_spacing / 2

    mic_positions = {
        # Outer perimeter - Bottom edge
        1: (0 - center_offset_x, 0 - center_offset_y),
        3: (outer_spacing - center_offset_x, 0 - center_offset_y),
        5: (2 * outer_spacing - center_offset_x, 0 - center_offset_y),
        7: (3 * outer_spacing - center_offset_x, 0 - center_offset_y),
        # Outer perimeter - Right edge
        6: (3 * outer_spacing - center_offset_x, outer_spacing - center_offset_y),
        9: (3 * outer_spacing - center_offset_x, 2 * outer_spacing - center_offset_y),
        # Outer perimeter - Top edge
        8: (3 * outer_spacing - center_offset_x, 3 * outer_spacing - center_offset_y),
        10: (2 * outer_spacing - center_offset_x, 3 * outer_spacing - center_offset_y),
        12: (outer_spacing - center_offset_x, 3 * outer_spacing - center_offset_y),
        14: (0 - center_offset_x, 3 * outer_spacing - center_offset_y),
        # Outer perimeter - Left edge
        15: (0 - center_offset_x, 2 * outer_spacing - center_offset_y),
        0: (0 - center_offset_x, outer_spacing - center_offset_y),
        # Inner square
        2: (-inner_half_spacing, -inner_half_spacing),
        4: (inner_half_spacing, -inner_half_spacing),
        11: (inner_half_spacing, inner_half_spacing),
        13: (-inner_half_spacing, inner_half_spacing)
    }
    return mic_positions


def simulate_array_signals(source_signal, source_azimuth_deg, source_distance,
                           snr_db=20, sample_rate=SAMPLE_RATE):
    """
    Simulate microphone array signals for a far-field source.

    Args:
        source_signal: Mono audio signal from the source
        source_azimuth_deg: Azimuth angle in degrees (0° = North, clockwise)
        source_distance: Distance to source in meters (for attenuation)
        snr_db: Signal-to-noise ratio in dB
        sample_rate: Sample rate in Hz

    Returns:
        mic_signals: Array of shape [samples, channels] with simulated signals
    """
    mic_positions = get_mic_positions_4x4()
    num_samples = len(source_signal)

    # Convert azimuth to radians (0° = North = +Y, 90° = East = +X)
    theta_rad = np.radians(90 - source_azimuth_deg)

    # Source direction unit vector (pointing FROM source TO array)
    source_direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    # Calculate signal power for SNR
    signal_power = np.mean(source_signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)

    # Generate signals for each microphone
    mic_signals = np.zeros((num_samples, CHANNELS))

    for ch in range(CHANNELS):
        pos = np.array(mic_positions[ch])

        # Time delay: positive when mic is closer to source
        # For far-field, delay = (mic_position · source_direction) / c
        distance_proj = np.dot(pos, source_direction)
        time_delay = distance_proj / SPEED_OF_SOUND
        delay_samples = int(time_delay * sample_rate)

        # Apply delay to signal
        if delay_samples > 0:
            # Mic is closer to source - signal arrives earlier
            delayed = np.pad(source_signal, (0, delay_samples), mode='constant')[delay_samples:]
        elif delay_samples < 0:
            # Mic is farther from source - signal arrives later
            delayed = np.pad(source_signal, (-delay_samples, 0), mode='constant')[:num_samples]
        else:
            delayed = source_signal.copy()

        # Ensure correct length
        if len(delayed) > num_samples:
            delayed = delayed[:num_samples]
        elif len(delayed) < num_samples:
            delayed = np.pad(delayed, (0, num_samples - len(delayed)), mode='constant')

        # Apply distance attenuation (1/r for pressure)
        # For simulation, we use relative attenuation within the array
        mic_distance = source_distance + distance_proj  # Approximate distance to each mic
        attenuation = source_distance / mic_distance  # Relative to center

        # Add noise
        noise = np.random.randn(num_samples) * noise_std

        mic_signals[:, ch] = delayed * attenuation + noise

    return mic_signals


def beamforming_scan(audio_data, azimuth_resolution=AZIMUTH_RESOLUTION):
    """
    Perform delay-and-sum beamforming scan across all azimuths.

    Args:
        audio_data: Audio data array [samples, channels]
        azimuth_resolution: Angular resolution in degrees

    Returns:
        azimuth_scan: Array of scanned azimuth angles
        power_spectrum: Normalized power at each azimuth
    """
    mic_positions = get_mic_positions_4x4()

    # Normalize signals
    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)

    azimuth_scan = np.arange(0, 360, azimuth_resolution)
    power_spectrum = np.zeros(len(azimuth_scan))

    num_samples = len(audio_data)

    for idx, azimuth in enumerate(azimuth_scan):
        theta_rad = np.radians(90 - azimuth)
        source_direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

        delayed_signals = []
        for ch in range(CHANNELS):
            pos = np.array(mic_positions[ch])
            distance_proj = np.dot(pos, source_direction)
            # Positive projection = mic closer to source = signal arrives earlier
            # We need to DELAY (shift right) to align, so positive delay for positive projection
            time_delay = distance_proj / SPEED_OF_SOUND
            delay_samples = int(time_delay * SAMPLE_RATE)

            if delay_samples > 0:
                delayed = np.pad(audio_data[:, ch], (delay_samples, 0), mode='constant')[:num_samples]
            elif delay_samples < 0:
                delayed = np.pad(audio_data[:, ch], (0, -delay_samples), mode='constant')[-num_samples:]
            else:
                delayed = audio_data[:, ch]

            delayed_signals.append(delayed)

        beamformed = np.mean(delayed_signals, axis=0)
        power_spectrum[idx] = np.mean(beamformed ** 2)

    # Normalize
    power_spectrum = power_spectrum / (np.max(power_spectrum) + 1e-10)

    return azimuth_scan, power_spectrum


def detect_peaks_no_snr(azimuth_scan, power_spectrum, min_separation=MIN_SEPARATION,
                        min_peak_height=MIN_PEAK_HEIGHT, min_prominence=MIN_PROMINENCE):
    """
    Detect peaks WITHOUT SNR filtering (original method).

    Returns:
        peaks: List of dictionaries with 'azimuth' and 'power'
    """
    azimuth_resolution = azimuth_scan[1] - azimuth_scan[0]

    peak_indices, properties = find_peaks(
        power_spectrum,
        height=min_peak_height,
        distance=int(min_separation / azimuth_resolution),
        prominence=min_prominence
    )

    peaks = []
    for idx in peak_indices:
        peaks.append({
            'azimuth': azimuth_scan[idx],
            'power': power_spectrum[idx],
            'snr_db': None  # Not calculated
        })

    # Sort by power
    peaks = sorted(peaks, key=lambda x: x['power'], reverse=True)

    return peaks


def detect_peaks_with_snr(azimuth_scan, power_spectrum, min_separation=MIN_SEPARATION,
                          min_peak_height=MIN_PEAK_HEIGHT, min_prominence=MIN_PROMINENCE,
                          snr_threshold_db=SNR_THRESHOLD_DB, noise_percentile=NOISE_PERCENTILE):
    """
    Detect peaks WITH SNR filtering.

    Uses the same base peak detection as the non-SNR method, but adds an
    additional SNR check. The noise floor is estimated as a percentile of
    the power spectrum. Only peaks that exceed the SNR threshold are accepted.

    Returns:
        peaks: List of dictionaries with 'azimuth', 'power', and 'snr_db'
        noise_floor: Estimated noise floor value
    """
    azimuth_resolution = azimuth_scan[1] - azimuth_scan[0]

    # Estimate noise floor as percentile of power spectrum
    noise_floor = np.percentile(power_spectrum, noise_percentile)

    # SNR threshold in linear scale (for reference)
    # snr_threshold_linear = 10 ** (snr_threshold_db / 10)

    # Find all local maxima using same parameters as non-SNR method
    peak_indices, properties = find_peaks(
        power_spectrum,
        height=min_peak_height,
        distance=int(min_separation / azimuth_resolution),
        prominence=min_prominence
    )

    peaks = []
    for idx in peak_indices:
        peak_power = power_spectrum[idx]

        # Calculate SNR for this peak
        if noise_floor > 0:
            snr_linear = peak_power / noise_floor
            snr_db = 10 * np.log10(snr_linear)
        else:
            snr_db = float('inf')

        # Only accept peaks that exceed SNR threshold
        if snr_db >= snr_threshold_db:
            peaks.append({
                'azimuth': azimuth_scan[idx],
                'power': peak_power,
                'snr_db': snr_db
            })

    # Sort by power
    peaks = sorted(peaks, key=lambda x: x['power'], reverse=True)

    return peaks, noise_floor


def run_comparison(audio_file=None):
    """
    Run the SNR filter comparison test.
    """
    print("=" * 70)
    print("SNR FILTER COMPARISON FOR BEAMFORMING")
    print("=" * 70)
    print(f"Source distance: {SOURCE_DISTANCE} m")
    print(f"Source azimuth: {SOURCE_AZIMUTH}°")
    print(f"Simulation SNR: {SNR_DB} dB")
    print(f"SNR filter threshold: {SNR_THRESHOLD_DB} dB")
    print("=" * 70)

    # Load or generate source signal
    if audio_file and os.path.exists(audio_file):
        print(f"\nLoading audio file: {audio_file}")
        fs, source_signal = wavfile.read(audio_file)

        # Convert to mono if stereo
        if len(source_signal.shape) > 1:
            source_signal = np.mean(source_signal, axis=1)

        # Convert to float and normalize
        source_signal = source_signal.astype(float)
        source_signal = source_signal / (np.max(np.abs(source_signal)) + 1e-10)

        # Resample if necessary
        if fs != SAMPLE_RATE:
            print(f"Resampling from {fs} Hz to {SAMPLE_RATE} Hz")
            from scipy.signal import resample
            num_samples = int(len(source_signal) * SAMPLE_RATE / fs)
            source_signal = resample(source_signal, num_samples)

        # Use 0.5 seconds of audio
        duration_samples = int(0.5 * SAMPLE_RATE)
        if len(source_signal) > duration_samples:
            source_signal = source_signal[:duration_samples]
    else:
        print("\nGenerating synthetic test signal (3 kHz tone)")
        duration = 0.5  # seconds
        t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False)
        source_signal = np.sin(2 * np.pi * 3000 * t)  # 3 kHz tone

    print(f"Signal duration: {len(source_signal) / SAMPLE_RATE:.3f} seconds")
    print(f"Signal samples: {len(source_signal)}")

    # Simulate array signals
    print(f"\nSimulating array signals...")
    mic_signals = simulate_array_signals(
        source_signal,
        SOURCE_AZIMUTH,
        SOURCE_DISTANCE,
        snr_db=SNR_DB
    )
    print(f"Generated {mic_signals.shape[1]} channel signals")

    # Run beamforming
    print("\nRunning beamforming scan...")
    azimuth_scan, power_spectrum = beamforming_scan(mic_signals)

    # Detect peaks without SNR filter
    print("\nDetecting peaks WITHOUT SNR filter...")
    peaks_no_snr = detect_peaks_no_snr(azimuth_scan, power_spectrum)
    print(f"  Found {len(peaks_no_snr)} peaks")
    for i, peak in enumerate(peaks_no_snr[:5]):  # Show top 5
        print(f"    Peak {i+1}: Azimuth = {peak['azimuth']:.1f}°, Power = {peak['power']:.3f}")

    # Detect peaks with SNR filter
    print(f"\nDetecting peaks WITH SNR filter (threshold = {SNR_THRESHOLD_DB} dB)...")
    peaks_with_snr, noise_floor = detect_peaks_with_snr(azimuth_scan, power_spectrum)
    print(f"  Noise floor estimate: {noise_floor:.4f}")
    print(f"  Found {len(peaks_with_snr)} peaks")
    for i, peak in enumerate(peaks_with_snr[:5]):  # Show top 5
        print(f"    Peak {i+1}: Azimuth = {peak['azimuth']:.1f}°, Power = {peak['power']:.3f}, SNR = {peak['snr_db']:.1f} dB")

    # Calculate error for detected peaks
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"True source azimuth: {SOURCE_AZIMUTH}°")

    if len(peaks_no_snr) > 0:
        best_no_snr = peaks_no_snr[0]
        error_no_snr = abs(best_no_snr['azimuth'] - SOURCE_AZIMUTH)
        if error_no_snr > 180:
            error_no_snr = 360 - error_no_snr
        print(f"\nWithout SNR filter:")
        print(f"  Detected azimuth: {best_no_snr['azimuth']:.1f}°")
        print(f"  Error: {error_no_snr:.1f}°")
        print(f"  Total peaks detected: {len(peaks_no_snr)}")

    if len(peaks_with_snr) > 0:
        best_with_snr = peaks_with_snr[0]
        error_with_snr = abs(best_with_snr['azimuth'] - SOURCE_AZIMUTH)
        if error_with_snr > 180:
            error_with_snr = 360 - error_with_snr
        print(f"\nWith SNR filter ({SNR_THRESHOLD_DB} dB threshold):")
        print(f"  Detected azimuth: {best_with_snr['azimuth']:.1f}°")
        print(f"  Error: {error_with_snr:.1f}°")
        print(f"  SNR: {best_with_snr['snr_db']:.1f} dB")
        print(f"  Total peaks detected: {len(peaks_with_snr)}")

    # Create comparison plot
    print("\nGenerating comparison plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Power spectrum without SNR filter
    ax1 = axes[0, 0]
    ax1.plot(azimuth_scan, power_spectrum, 'b-', linewidth=1.5, label='Power Spectrum')
    ax1.axhline(y=MIN_PEAK_HEIGHT, color='gray', linestyle=':', label=f'Threshold ({MIN_PEAK_HEIGHT})')
    ax1.axvline(x=SOURCE_AZIMUTH, color='green', linestyle='--', linewidth=2, label=f'True Source ({SOURCE_AZIMUTH}°)')

    # Mark detected peaks
    for i, peak in enumerate(peaks_no_snr[:3]):
        color = 'red' if i == 0 else 'orange'
        ax1.axvline(x=peak['azimuth'], color=color, linestyle='-', alpha=0.7)
        ax1.plot(peak['azimuth'], peak['power'], 'o', color=color, markersize=10)
        ax1.annotate(f"{peak['azimuth']:.0f}°", (peak['azimuth'], peak['power'] + 0.05),
                    ha='center', fontsize=9, color=color)

    ax1.set_xlabel('Azimuth (degrees)')
    ax1.set_ylabel('Normalized Power')
    ax1.set_title(f'WITHOUT SNR Filter\n{len(peaks_no_snr)} peaks detected', fontsize=12, weight='bold')
    ax1.set_xlim(0, 360)
    ax1.set_ylim(0, 1.2)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])

    # Plot 2: Power spectrum with SNR filter
    ax2 = axes[0, 1]
    ax2.plot(azimuth_scan, power_spectrum, 'b-', linewidth=1.5, label='Power Spectrum')
    ax2.axhline(y=noise_floor, color='purple', linestyle='--', label=f'Noise Floor ({noise_floor:.3f})')

    # SNR threshold line
    snr_threshold_power = noise_floor * (10 ** (SNR_THRESHOLD_DB / 10))
    ax2.axhline(y=snr_threshold_power, color='red', linestyle=':',
                label=f'SNR Threshold ({SNR_THRESHOLD_DB} dB)')
    ax2.axvline(x=SOURCE_AZIMUTH, color='green', linestyle='--', linewidth=2,
                label=f'True Source ({SOURCE_AZIMUTH}°)')

    # Mark detected peaks
    for i, peak in enumerate(peaks_with_snr[:3]):
        color = 'red' if i == 0 else 'orange'
        ax2.axvline(x=peak['azimuth'], color=color, linestyle='-', alpha=0.7)
        ax2.plot(peak['azimuth'], peak['power'], 'o', color=color, markersize=10)
        ax2.annotate(f"{peak['azimuth']:.0f}°\n({peak['snr_db']:.1f}dB)",
                    (peak['azimuth'], peak['power'] + 0.05),
                    ha='center', fontsize=9, color=color)

    ax2.set_xlabel('Azimuth (degrees)')
    ax2.set_ylabel('Normalized Power')
    ax2.set_title(f'WITH SNR Filter ({SNR_THRESHOLD_DB} dB threshold)\n{len(peaks_with_snr)} peaks detected',
                  fontsize=12, weight='bold')
    ax2.set_xlim(0, 360)
    ax2.set_ylim(0, 1.2)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])

    # Plot 3: Polar comparison
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)

    # True source direction
    true_rad = np.radians(SOURCE_AZIMUTH)
    ax3.arrow(true_rad, 0, 0, 0.8, head_width=0.15, head_length=0.08,
              fc='green', ec='green', linewidth=3, label='True Source')

    # Detected directions (without SNR)
    for i, peak in enumerate(peaks_no_snr[:3]):
        az_rad = np.radians(peak['azimuth'])
        length = 0.3 + 0.4 * peak['power']
        color = 'red' if i == 0 else 'lightcoral'
        ax3.arrow(az_rad, 0, 0, length, head_width=0.12, head_length=0.06,
                  fc=color, ec=color, linewidth=2, alpha=0.7)

    ax3.set_ylim(0, 1)
    ax3.set_title('Polar View - Without SNR Filter', fontsize=11, weight='bold', pad=20)
    ax3.legend(['True', 'Detected'], loc='lower right')

    # Plot 4: Polar comparison with SNR
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    ax4.set_theta_zero_location('N')
    ax4.set_theta_direction(-1)

    # True source direction
    ax4.arrow(true_rad, 0, 0, 0.8, head_width=0.15, head_length=0.08,
              fc='green', ec='green', linewidth=3, label='True Source')

    # Detected directions (with SNR)
    for i, peak in enumerate(peaks_with_snr[:3]):
        az_rad = np.radians(peak['azimuth'])
        length = 0.3 + 0.4 * peak['power']
        color = 'blue' if i == 0 else 'lightblue'
        ax4.arrow(az_rad, 0, 0, length, head_width=0.12, head_length=0.06,
                  fc=color, ec=color, linewidth=2, alpha=0.7)

    ax4.set_ylim(0, 1)
    ax4.set_title('Polar View - With SNR Filter', fontsize=11, weight='bold', pad=20)
    ax4.legend(['True', 'Detected'], loc='lower right')

    plt.suptitle(f'SNR Filter Comparison\nSource: {SOURCE_DISTANCE}m @ {SOURCE_AZIMUTH}°, Simulation SNR: {SNR_DB} dB',
                 fontsize=14, weight='bold', y=1.02)

    plt.tight_layout()

    # Save figure
    output_file = 'snr_filter_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    plt.show()

    return peaks_no_snr, peaks_with_snr, noise_floor


if __name__ == "__main__":
    # Try to use the drone audio file, fall back to synthetic signal
    audio_files = [
        "../drone_monotone_3kHz.wav",
        "../Binary_Drone_Audio_yes_drone_B_S2_D1_067-bebop_000_.wav",
    ]

    audio_file = None
    for f in audio_files:
        if os.path.exists(f):
            audio_file = f
            break

    run_comparison(audio_file)
