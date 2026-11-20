"""
Experiment 2: Multiple Stationary Sources

Three drone audio sources at 30deg, 45deg, 60deg (15deg increments), all at 10m.
Each source uses the same real drone WAV recording.
Generates power spectrum + polar view (1x2 figure).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from sim_common import (
    load_source_wav, simulate_array_signals, beamforming_scan, detect_peaks,
    SOURCE_AZIMUTH_INITIAL, SOURCE_DISTANCE, SNR_DB, MIC_SPACING,
    MULTI_SOURCE_INCREMENT, NUM_SOURCES, CHANNELS, SAMPLE_RATE
)


def main():
    np.random.seed(42)

    source_azimuths = [SOURCE_AZIMUTH_INITIAL + i * MULTI_SOURCE_INCREMENT
                       for i in range(NUM_SOURCES)]

    print("=" * 60)
    print("EXPERIMENT 2: Multiple Stationary Sources")
    print(f"  Azimuths = {source_azimuths}, Distance = {SOURCE_DISTANCE}m")
    print(f"  SNR = {SNR_DB}dB, Mic spacing = {MIC_SPACING*1000:.0f}mm")
    print("=" * 60)

    # Load different segments of the drone WAV for each source (incoherent)
    duration_per_source = 0.5
    _, full_signal = load_source_wav(duration=duration_per_source * (NUM_SOURCES + 1))
    samples_per = int(duration_per_source * SAMPLE_RATE)

    mic_signals_total = np.zeros((samples_per, CHANNELS))
    for i, az in enumerate(source_azimuths):
        seg_start = i * (samples_per // 2)  # overlapping segments for variety
        source_segment = full_signal[seg_start:seg_start + samples_per]
        if len(source_segment) < samples_per:
            source_segment = np.pad(source_segment,
                                    (0, samples_per - len(source_segment)))
        mic_signals_total += simulate_array_signals(source_segment, az,
                                                    SOURCE_DISTANCE)

    azimuth_scan, power_spectrum = beamforming_scan(mic_signals_total)
    peaks, noise_floor = detect_peaks(azimuth_scan, power_spectrum)

    print(f"\n  Detected {len(peaks)} peak(s)")
    for p in peaks:
        errors = [abs(p['azimuth'] - az) for az in source_azimuths]
        errors = [e if e <= 180 else 360 - e for e in errors]
        min_err = min(errors)
        print(f"    -> {p['azimuth']:.0f}deg (closest source error {min_err:.1f}deg, "
              f"SNR {p['snr_db']:.1f}dB)")

    # ── Plot: 1x2 figure ───────────────────────────────────────────────
    colors_src = ['green', 'blue', 'purple']

    fig, (ax2, _) = plt.subplots(1, 2, figsize=(14, 5))
    _.remove()
    ax2p = fig.add_subplot(1, 2, 2, projection='polar')

    # Power spectrum
    ax2.plot(azimuth_scan, power_spectrum, 'b-', linewidth=1.5,
             label='Power spectrum')
    for i, az in enumerate(source_azimuths):
        ax2.axvline(x=az, color=colors_src[i], linestyle='--', linewidth=2,
                    label=f'Source {i+1} ({az:.0f}deg)')
    for p in peaks:
        ax2.plot(p['azimuth'], p['power'], 'ro', markersize=10)
        ax2.annotate(f"{p['azimuth']:.0f}deg",
                     (p['azimuth'], p['power'] + 0.04),
                     ha='center', fontsize=9, color='red')
    ax2.set_xlabel('Azimuth (degrees)')
    ax2.set_ylabel('Normalized Power')
    ax2.set_title(f'{NUM_SOURCES} Stationary Sources - Power Spectrum',
                  fontsize=12, weight='bold')
    ax2.set_xlim(0, 360)
    ax2.set_ylim(0, 1.3)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(np.arange(0, 361, 45))

    # Polar view
    ax2p.set_theta_zero_location('N')
    ax2p.set_theta_direction(-1)
    ax2p.plot(np.radians(azimuth_scan), power_spectrum, 'b-', linewidth=1)
    for i, az in enumerate(source_azimuths):
        ax2p.arrow(np.radians(az), 0, 0, 0.85,
                   head_width=0.1, head_length=0.06,
                   fc=colors_src[i], ec=colors_src[i], linewidth=2)
    for p in peaks:
        ax2p.plot(np.radians(p['azimuth']), p['power'], 'ro', markersize=8)
    ax2p.set_title('Polar View', fontsize=12, weight='bold', pad=20)
    ax2p.set_ylim(0, 1.1)

    fig.suptitle(
        f'Experiment 2: {NUM_SOURCES} Stationary Sources (Drone Audio)\n'
        f'4x4 Array, {SOURCE_DISTANCE}m, SNR={SNR_DB}dB',
        fontsize=13, weight='bold', y=1.02)

    plt.tight_layout()
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'exp2_multi_stationary.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  Saved plot to: {output_file}")
    plt.show()


if __name__ == "__main__":
    main()
