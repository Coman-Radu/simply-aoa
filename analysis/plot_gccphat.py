import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Constants for DOA calculation
SPEED_OF_SOUND = 343.0  # m/s at room temperature
MIC_SPACING = 0.042  # meters (spacing between mics in the 4x4 array)

# Audio settings
FFT_SIZE = 2048  # FFT size for GCC-PHAT

# Known source direction for this example
SOURCE_AZIMUTH = 30.0  # degrees (0° = North, clockwise)


def get_mic_positions_4x4():
    """
    Generate microphone positions for 4x4 array with origin at CENTER of array.

    Returns:
        Dictionary mapping channel numbers to (x, y) positions in meters
    """
    spacing = MIC_SPACING

    # X positions (West to East)
    x_positions = [-1.5 * spacing, -0.5 * spacing, 0.5 * spacing, 1.5 * spacing]

    # Y positions (South to North)
    y_positions = [-1.5 * spacing, -0.5 * spacing, 0.5 * spacing, 1.5 * spacing]

    # Build position dictionary for each channel
    mic_positions = {}

    # Row 0 (South, Y=-63mm): channels [2, 1, 0, 15]
    mic_positions[2] = (x_positions[0], y_positions[0])  # MIC03
    mic_positions[1] = (x_positions[1], y_positions[0])  # MIC02
    mic_positions[0] = (x_positions[2], y_positions[0])  # MIC01
    mic_positions[15] = (x_positions[3], y_positions[0])  # MIC16

    # Row 1 (Y=-21mm): channels [4, 3, 14, 13]
    mic_positions[4] = (x_positions[0], y_positions[1])  # MIC05
    mic_positions[3] = (x_positions[1], y_positions[1])  # MIC04
    mic_positions[14] = (x_positions[2], y_positions[1])  # MIC15
    mic_positions[13] = (x_positions[3], y_positions[1])  # MIC14

    # Row 2 (Y=+21mm): channels [6, 5, 12, 11]
    mic_positions[6] = (x_positions[0], y_positions[2])  # MIC07
    mic_positions[5] = (x_positions[1], y_positions[2])  # MIC06
    mic_positions[12] = (x_positions[2], y_positions[2])  # MIC13
    mic_positions[11] = (x_positions[3], y_positions[2])  # MIC12

    # Row 3 (North, Y=+63mm): channels [8, 7, 10, 9]
    mic_positions[8] = (x_positions[0], y_positions[3])  # MIC09
    mic_positions[7] = (x_positions[1], y_positions[3])  # MIC08
    mic_positions[10] = (x_positions[2], y_positions[3])  # MIC11
    mic_positions[9] = (x_positions[3], y_positions[3])  # MIC10

    return mic_positions


def gcc_phat(sig1, sig2, fs, max_tau=None):
    """
    Compute GCC-PHAT (Generalized Cross-Correlation with Phase Transform).

    Args:
        sig1: First signal
        sig2: Second signal
        fs: Sample rate
        max_tau: Maximum time delay in samples (for limiting search range)

    Returns:
        tau: Time delay in samples (sig2 relative to sig1)
        gcc: GCC-PHAT function values
        lags: Corresponding lag values in samples
    """
    # Pad signals to FFT size
    n = FFT_SIZE

    # FFT of both signals
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)

    # Cross-power spectrum
    R = SIG1 * np.conj(SIG2)

    # PHAT weighting: normalize by magnitude
    R_phat = R / (np.abs(R) + 1e-10)

    # Inverse FFT to get GCC
    gcc = np.fft.irfft(R_phat, n=n)

    # Shift to center (make symmetric around zero)
    gcc = np.fft.fftshift(gcc)

    # Create lag array
    lags = np.arange(-len(gcc) // 2, len(gcc) // 2)

    # Limit search range if specified
    if max_tau is not None:
        center = len(gcc) // 2
        gcc_limited = np.zeros_like(gcc)
        start = max(0, center - max_tau)
        end = min(len(gcc), center + max_tau + 1)
        gcc_limited[start:end] = gcc[start:end]
        gcc = gcc_limited

    # Find peak
    peak_idx = np.argmax(gcc)
    tau = lags[peak_idx]

    return tau, gcc, lags


def compute_expected_delay(ch1, ch2, azimuth, mic_positions, fs):
    """
    Compute expected time delay for a source at given azimuth.

    Args:
        ch1: First microphone channel
        ch2: Second microphone channel
        azimuth: Source azimuth in degrees (0° = North, clockwise)
        mic_positions: Dictionary of microphone positions
        fs: Sample rate

    Returns:
        Expected time delay in samples
    """
    # Convert azimuth to radians (0° = North, 90° = East)
    theta_rad = np.radians(90 - azimuth)

    # Source direction unit vector
    source_direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    # Get positions
    pos1 = np.array(mic_positions[ch1])
    pos2 = np.array(mic_positions[ch2])

    # Expected time delay
    distance_diff = np.dot(pos2 - pos1, source_direction)
    expected_tau = -distance_diff * fs / SPEED_OF_SOUND

    return expected_tau


def select_interesting_pairs(mic_positions, azimuth, num_pairs=6):
    """
    Select interesting microphone pairs for visualization.
    Choose pairs at different orientations relative to source direction.

    Args:
        mic_positions: Dictionary of microphone positions
        azimuth: Source azimuth in degrees
        num_pairs: Number of pairs to select

    Returns:
        List of (ch1, ch2, description) tuples
    """
    # Convert azimuth to radians
    theta_rad = np.radians(90 - azimuth)
    source_direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    # Calculate alignment scores for all pairs
    pair_info = []
    for ch1 in range(16):
        for ch2 in range(ch1 + 1, 16):
            pos1 = np.array(mic_positions[ch1])
            pos2 = np.array(mic_positions[ch2])

            # Pair baseline vector
            baseline = pos2 - pos1
            baseline_norm = np.linalg.norm(baseline)

            if baseline_norm > 1e-6:
                baseline_unit = baseline / baseline_norm

                # Alignment with source direction
                alignment = np.dot(baseline_unit, source_direction)

                # Distance
                distance = baseline_norm

                pair_info.append((ch1, ch2, alignment, distance))

    # Sort by alignment (want variety: parallel, perpendicular, diagonal)
    pair_info.sort(key=lambda x: x[2])  # Sort by alignment

    # Select diverse pairs
    selected = []

    # Perpendicular pair (alignment near 0)
    for ch1, ch2, align, dist in pair_info:
        if abs(align) < 0.3 and dist > MIC_SPACING * 0.5:
            selected.append((ch1, ch2, f"CH{ch1+1}-CH{ch2+1} (perpendicular)"))
            break

    # Parallel pair (alignment near 1)
    for ch1, ch2, align, dist in reversed(pair_info):
        if align > 0.7 and dist > MIC_SPACING * 0.5:
            selected.append((ch1, ch2, f"CH{ch1+1}-CH{ch2+1} (aligned)"))
            break

    # Anti-parallel pair (alignment near -1)
    for ch1, ch2, align, dist in pair_info:
        if align < -0.7 and dist > MIC_SPACING * 0.5:
            selected.append((ch1, ch2, f"CH{ch1+1}-CH{ch2+1} (opposite)"))
            break

    # Diagonal pairs (various alignments)
    for ch1, ch2, align, dist in pair_info:
        if 0.3 < abs(align) < 0.7 and dist > MIC_SPACING * 1.5:
            if len(selected) < num_pairs:
                selected.append((ch1, ch2, f"CH{ch1+1}-CH{ch2+1} (diagonal)"))

    # Add more pairs to reach num_pairs
    for ch1, ch2, align, dist in pair_info:
        if len(selected) >= num_pairs:
            break
        if (ch1, ch2, f"CH{ch1+1}-CH{ch2+1}") not in [(s[0], s[1], s[2]) for s in selected]:
            selected.append((ch1, ch2, f"CH{ch1+1}-CH{ch2+1}"))

    return selected[:num_pairs]


if __name__ == "__main__":
    print("=" * 80)
    print("GCC-PHAT CROSS-CORRELATION VISUALIZATION")
    print("=" * 80)
    print(f"Source azimuth: {SOURCE_AZIMUTH}° (0° = North, clockwise)")
    print(f"Microphone spacing: {MIC_SPACING*1000:.1f} mm")
    print(f"FFT size: {FFT_SIZE}")
    print("=" * 80)
    print()

    # Load audio file
    audio_file = "Binary_Drone_Audio_yes_drone_B_S2_D1_067-bebop_000_.wav"

    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found!")
        print("Make sure the file is in the current directory.")
        exit(1)

    print(f"Loading audio file: {audio_file}")
    fs, audio_mono = wavfile.read(audio_file)
    print(f"Sample rate: {fs} Hz")
    print(f"Original audio shape: {audio_mono.shape}")
    print(f"Duration: {len(audio_mono)/fs:.2f} seconds")

    # Get microphone positions
    mic_positions = get_mic_positions_4x4()

    # Simulate 16-channel array by adding time delays to mono signal
    # based on source direction at SOURCE_AZIMUTH
    print(f"\nSimulating 16-channel array from mono source...")
    print(f"Simulated source direction: {SOURCE_AZIMUTH}°")

    # Convert azimuth to radians (0° = North, 90° = East)
    theta_rad = np.radians(90 - SOURCE_AZIMUTH)
    source_direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    # Create multi-channel array
    audio_data = np.zeros((len(audio_mono), 16))

    for ch in range(16):
        pos = np.array(mic_positions[ch])

        # Time delay for this mic (negative because signal arrives earlier if mic is closer)
        distance_proj = np.dot(pos, source_direction)
        time_delay = -distance_proj / SPEED_OF_SOUND
        delay_samples = int(time_delay * fs)

        # Apply delay to signal
        if delay_samples > 0:
            # Pad at beginning, truncate at end
            audio_data[:, ch] = np.pad(audio_mono, (delay_samples, 0), mode='constant')[:len(audio_mono)]
        elif delay_samples < 0:
            # Pad at end, truncate at beginning
            audio_data[:, ch] = np.pad(audio_mono, (0, -delay_samples), mode='constant')[-len(audio_mono):]
        else:
            audio_data[:, ch] = audio_mono

    print(f"Created 16-channel array with shape: {audio_data.shape}")
    print()

    # Select interesting pairs
    print(f"Selecting microphone pairs for visualization...")
    pairs = select_interesting_pairs(mic_positions, SOURCE_AZIMUTH, num_pairs=6)

    print(f"\nSelected {len(pairs)} microphone pairs:")
    for ch1, ch2, desc in pairs:
        pos1 = mic_positions[ch1]
        pos2 = mic_positions[ch2]
        distance = np.linalg.norm(np.array(pos2) - np.array(pos1))
        print(f"  {desc}: {distance*1000:.1f} mm apart")

    # Process a segment of audio
    segment_duration = 0.5  # seconds
    segment_samples = int(segment_duration * fs)
    start_sample = len(audio_data) // 2  # Use middle section

    audio_segment = audio_data[start_sample:start_sample + segment_samples]

    # Normalize
    audio_segment = audio_segment / (np.max(np.abs(audio_segment)) + 1e-10)

    print(f"\nProcessing {segment_duration} second segment...")
    print()

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'GCC-PHAT Cross-Correlation Curves\nSource at {SOURCE_AZIMUTH}° Azimuth',
                 fontsize=14, weight='bold')

    axes = axes.flatten()

    # Process each pair
    for idx, (ch1, ch2, desc) in enumerate(pairs):
        ax = axes[idx]

        # Extract signals
        sig1 = audio_segment[:, ch1]
        sig2 = audio_segment[:, ch2]

        # Compute GCC-PHAT
        measured_tau, gcc, lags = gcc_phat(sig1, sig2, fs, max_tau=30)

        # Compute expected delay
        expected_tau = compute_expected_delay(ch1, ch2, SOURCE_AZIMUTH, mic_positions, fs)

        # Convert lags to milliseconds for plotting
        lags_ms = lags / fs * 1000

        # Plot GCC-PHAT curve
        ax.plot(lags_ms, gcc, 'b-', linewidth=1.5, label='GCC-PHAT')

        # Mark measured peak
        ax.axvline(x=measured_tau/fs*1000, color='red', linestyle='--',
                   linewidth=2, label=f'Measured: {measured_tau/fs*1000:.3f} ms')

        # Mark expected delay
        ax.axvline(x=expected_tau/fs*1000, color='green', linestyle=':',
                   linewidth=2, label=f'Expected: {expected_tau/fs*1000:.3f} ms')

        # Annotate error
        error_samples = measured_tau - expected_tau
        error_ms = error_samples / fs * 1000
        error_text = f'Error: {error_ms:.3f} ms ({error_samples:.1f} samples)'

        ax.text(0.02, 0.98, error_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Labels and title
        ax.set_xlabel('Time Delay (ms)', fontsize=10)
        ax.set_ylabel('GCC-PHAT Value', fontsize=10)
        ax.set_title(desc, fontsize=11, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        # Zoom to relevant range
        ax.set_xlim(-0.6, 0.6)

        print(f"{desc}:")
        print(f"  Measured delay: {measured_tau/fs*1000:.3f} ms ({measured_tau:.1f} samples)")
        print(f"  Expected delay: {expected_tau/fs*1000:.3f} ms ({expected_tau:.1f} samples)")
        print(f"  Error: {error_ms:.3f} ms ({error_samples:.1f} samples)")
        print()

    plt.tight_layout()

    # Add array geometry visualization
    fig2, ax_array = plt.subplots(1, 1, figsize=(8, 8))
    fig2.suptitle(f'Microphone Array Geometry\nSource at {SOURCE_AZIMUTH}°',
                  fontsize=14, weight='bold')

    # Plot all microphones
    for ch, pos in mic_positions.items():
        ax_array.scatter(pos[0]*1000, pos[1]*1000, c='blue', s=100, alpha=0.6, zorder=3)
        ax_array.text(pos[0]*1000, pos[1]*1000 + 5, f'{ch+1}',
                     ha='center', va='bottom', fontsize=8)

    # Plot selected pairs
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
    for idx, (ch1, ch2, desc) in enumerate(pairs):
        pos1 = np.array(mic_positions[ch1]) * 1000
        pos2 = np.array(mic_positions[ch2]) * 1000

        color = colors[idx % len(colors)]
        ax_array.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                     color=color, linewidth=2, alpha=0.7, label=desc, zorder=2)

    # Draw source direction
    theta_rad = np.radians(90 - SOURCE_AZIMUTH)
    source_dir = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    arrow_length = 80  # mm
    ax_array.arrow(0, 0, source_dir[0]*arrow_length, source_dir[1]*arrow_length,
                  head_width=10, head_length=10, fc='red', ec='red',
                  linewidth=3, alpha=0.8, zorder=4)
    ax_array.text(source_dir[0]*arrow_length*1.2, source_dir[1]*arrow_length*1.2,
                 f'Source\n{SOURCE_AZIMUTH}°',
                 ha='center', va='center', fontsize=12, weight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Draw compass directions
    ax_array.text(0, 100, 'N', ha='center', va='bottom', fontsize=12, weight='bold')
    ax_array.text(100, 0, 'E', ha='left', va='center', fontsize=12, weight='bold')
    ax_array.text(0, -100, 'S', ha='center', va='top', fontsize=12, weight='bold')
    ax_array.text(-100, 0, 'W', ha='right', va='center', fontsize=12, weight='bold')

    ax_array.set_xlabel('X (East) [mm]', fontsize=11)
    ax_array.set_ylabel('Y (North) [mm]', fontsize=11)
    ax_array.set_aspect('equal')
    ax_array.grid(True, alpha=0.3)
    ax_array.legend(loc='upper right', fontsize=9)
    ax_array.set_xlim(-120, 120)
    ax_array.set_ylim(-120, 120)

    plt.tight_layout()

    print("=" * 80)
    print("Plots generated successfully!")
    print("Close the plot windows to exit.")
    print("=" * 80)

    plt.show()
