"""
Experiment 1: Single Stationary Source

Single drone audio source at 30deg, 10m from array center.
Uses real drone WAV recording as the source signal.
Generates power spectrum + polar view (1x2 figure).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from sim_common import (
    load_source_wav, simulate_array_signals, beamforming_scan, detect_peaks,
    SOURCE_AZIMUTH_INITIAL, SOURCE_DISTANCE, SNR_DB, MIC_SPACING
)


def main():
    np.random.seed(42)

    print("=" * 60)
    print("EXPERIMENT 1: Single Stationary Source")
    print(f"  Azimuth = {SOURCE_AZIMUTH_INITIAL}deg, Distance = {SOURCE_DISTANCE}m")
    print(f"  SNR = {SNR_DB}dB, Mic spacing = {MIC_SPACING*1000:.0f}mm")
    print("=" * 60)

    # Load drone WAV as source signal
    _, source_signal = load_source_wav(duration=0.5)
    mic_signals = simulate_array_signals(source_signal, SOURCE_AZIMUTH_INITIAL,
                                         SOURCE_DISTANCE)
    azimuth_scan, power_spectrum = beamforming_scan(mic_signals)
    peaks, noise_floor = detect_peaks(azimuth_scan, power_spectrum)

    print(f"\n  Detected {len(peaks)} peak(s)")
    for p in peaks:
        error = abs(p['azimuth'] - SOURCE_AZIMUTH_INITIAL)
        if error > 180:
            error = 360 - error
        print(f"    -> {p['azimuth']:.0f}deg (error {error:.1f}deg, "
              f"SNR {p['snr_db']:.1f}dB)")

    # ── Plot: 1x2 figure ───────────────────────────────────────────────
    fig, (ax1, ax1p) = plt.subplots(1, 2, figsize=(14, 5),
                                     subplot_kw={'projection': None})
    # Remove the second axes and recreate as polar
    ax1p.remove()
    ax1p = fig.add_subplot(1, 2, 2, projection='polar')

    # Power spectrum
    ax1.plot(azimuth_scan, power_spectrum, 'b-', linewidth=1.5,
             label='Power spectrum')
    ax1.axvline(x=SOURCE_AZIMUTH_INITIAL, color='green', linestyle='--',
                linewidth=2, label=f'True source ({SOURCE_AZIMUTH_INITIAL}deg)')
    for p in peaks:
        ax1.plot(p['azimuth'], p['power'], 'ro', markersize=10)
        ax1.annotate(f"{p['azimuth']:.0f}deg\n({p['snr_db']:.1f}dB)",
                     (p['azimuth'], p['power'] + 0.04),
                     ha='center', fontsize=9, color='red')
    ax1.set_xlabel('Azimuth (degrees)')
    ax1.set_ylabel('Normalized Power')
    ax1.set_title('Single Stationary Source - Power Spectrum',
                  fontsize=12, weight='bold')
    ax1.set_xlim(0, 360)
    ax1.set_ylim(0, 1.3)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(np.arange(0, 361, 45))

    # Polar view
    ax1p.set_theta_zero_location('N')
    ax1p.set_theta_direction(-1)
    ax1p.plot(np.radians(azimuth_scan), power_spectrum, 'b-', linewidth=1)
    ax1p.arrow(np.radians(SOURCE_AZIMUTH_INITIAL), 0, 0, 0.85,
               head_width=0.1, head_length=0.06, fc='green', ec='green',
               linewidth=2)
    for p in peaks:
        ax1p.plot(np.radians(p['azimuth']), p['power'], 'ro', markersize=8)
    ax1p.set_title('Polar View', fontsize=12, weight='bold', pad=20)
    ax1p.set_ylim(0, 1.1)

    fig.suptitle(
        f'Experiment 1: Single Stationary Source (Drone Audio)\n'
        f'4x4 Array, {SOURCE_DISTANCE}m, SNR={SNR_DB}dB',
        fontsize=13, weight='bold', y=1.02)

    plt.tight_layout()
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'exp1_single_stationary.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  Saved plot to: {output_file}")
    plt.show()


if __name__ == "__main__":
    main()
