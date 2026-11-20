import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.io.wavfile import read
from dataclasses import dataclass

@dataclass
class MicArrayConfig:
    """Configuration for microphone array"""
    frequency: float = 3500.0  # Hz
    speed_of_sound: float = 343.0  # m/s in air at 20°C
    N: int = 4  # NxN array size

    @property
    def rows(self):
        """Number of rows"""
        return self.N

    @property
    def cols(self):
        """Number of columns"""
        return self.N

    @property
    def wavelength(self):
        """Calculate wavelength"""
        return self.speed_of_sound / self.frequency

    @property
    def spacing(self):
        """Calculate spacing (lambda/2)"""
        return self.wavelength / 2


def generate_mic_positions(config: MicArrayConfig):
    """Generate microphone positions in a grid"""
    spacing = config.spacing

    # Create grid positions centered at origin
    x_positions = np.arange(config.cols) * spacing
    y_positions = np.arange(config.rows) * spacing

    # Center the array at origin
    x_positions -= x_positions.mean()
    y_positions -= y_positions.mean()

    # Create mesh grid
    X, Y = np.meshgrid(x_positions, y_positions)

    # Flatten to get individual mic coordinates
    mic_positions = np.column_stack((X.flatten(), Y.flatten()))

    return mic_positions


def apply_trig_angle(delta_t, wavelength, speed_of_sound):
    """
    Calculate angle of arrival from time delay between microphones

    theta = arcsin((c * delta_t) / d)
    where d = lambda/2 = spacing
    """
    theta = np.arcsin((speed_of_sound * delta_t) / (wavelength / 2))
    return theta


def bandpass(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


if __name__ == "__main__":
    # Configuration
    config = MicArrayConfig(frequency=3500, N=4)

    print(f"Wavelength (lambda): {config.wavelength*1000:.2f} mm")
    print(f"Mic spacing (lambda/2): {config.spacing*1000:.2f} mm")
    print(f"Array size: {config.N}x{config.N} = {config.N**2} microphones")

    # Generate microphone positions
    mic_positions = generate_mic_positions(config)

    # Load reference WAV file
    fs, wav_data = read('drone_monotone_3kHz.wav')

    # Time vector
    t = np.arange(len(wav_data)) / fs

    # Create NxN microphone signals (for now, all same signal)
    num_mics = config.N * config.N
    mic_signals = np.zeros((num_mics, len(wav_data)))

    for i in range(num_mics):
        mic_signals[i] = wav_data
        # Apply bandpass filter
        mic_signals[i] = bandpass(mic_signals[i], 3000, 4000, fs)

    # Compute analytic signals and instantaneous phases
    analytic_signals = np.zeros_like(mic_signals, dtype=complex)
    phases = np.zeros_like(mic_signals)

    for i in range(num_mics):
        analytic_signals[i] = hilbert(mic_signals[i])
        phases[i] = np.unwrap(np.angle(analytic_signals[i]))

    # Use peak of envelope as reference point
    envelope_ref = np.abs(analytic_signals[0])
    peak_idx = np.argmax(envelope_ref)

    print(f"\nReference peak at sample {peak_idx} (time = {peak_idx/fs*1e3:.2f} ms)")

    # Calculate phase differences and time delays for X-axis (row-wise)
    print("\n" + "="*60)
    print("X-AXIS (AZIMUTH) - ROW-WISE ANALYSIS")
    print("="*60)

    # Analyze first row (mics 0, 1, 2, 3)
    row_0_indices = list(range(config.N))
    phase_diffs_x = []
    time_delays_x = []

    for j in range(len(row_0_indices) - 1):
        mic_a = row_0_indices[j]
        mic_b = row_0_indices[j + 1]

        # Phase difference at peak
        delta_phi = phases[mic_b, peak_idx] - phases[mic_a, peak_idx]

        # Convert to time delay: τ = Δφ / (2πf)
        tau = delta_phi / (2 * np.pi * config.frequency)

        phase_diffs_x.append(delta_phi)
        time_delays_x.append(tau)

        print(f"  Mic {mic_a+1} to Mic {mic_b+1}:")
        print(f"    Phase diff: {np.degrees(delta_phi):.2f} deg")
        print(f"    Time delay: {tau*1e6:.2f} us")

    # Average time delay
    tau_avg_x = np.mean(time_delays_x)

    # Calculate angle of arrival using trig formula
    if abs(config.speed_of_sound * tau_avg_x / config.spacing) <= 1.0:
        theta_x = apply_trig_angle(tau_avg_x, config.wavelength, config.speed_of_sound)
        print(f"\nAverage time delay (X): {tau_avg_x*1e6:.2f} us")
        print(f"Angle of arrival (X-axis): {np.degrees(theta_x):.2f} deg")
    else:
        print(f"\nWarning: c·τ/d = {config.speed_of_sound * tau_avg_x / config.spacing:.3f} exceeds [-1, 1]")
        print("Cannot calculate angle (may indicate noise or incorrect parameters)")

    # Calculate phase differences and time delays for Y-axis (column-wise)
    print("\n" + "="*60)
    print("Y-AXIS (ELEVATION) - COLUMN-WISE ANALYSIS")
    print("="*60)

    # Analyze first column (mics 0, N, 2N, 3N)
    col_0_indices = list(range(0, num_mics, config.N))
    phase_diffs_y = []
    time_delays_y = []

    for j in range(len(col_0_indices) - 1):
        mic_a = col_0_indices[j]
        mic_b = col_0_indices[j + 1]

        # Phase difference at peak
        delta_phi = phases[mic_b, peak_idx] - phases[mic_a, peak_idx]

        # Convert to time delay: τ = Δφ / (2πf)
        tau = delta_phi / (2 * np.pi * config.frequency)

        phase_diffs_y.append(delta_phi)
        time_delays_y.append(tau)

        print(f"  Mic {mic_a+1} to Mic {mic_b+1}:")
        print(f"    Phase diff: {np.degrees(delta_phi):.2f} deg")
        print(f"    Time delay: {tau*1e6:.2f} us")

    # Average time delay
    tau_avg_y = np.mean(time_delays_y)

    # Calculate angle of arrival using trig formula
    if abs(config.speed_of_sound * tau_avg_y / config.spacing) <= 1.0:
        theta_y = apply_trig_angle(tau_avg_y, config.wavelength, config.speed_of_sound)
        print(f"\nAverage time delay (Y): {tau_avg_y*1e6:.2f} us")
        print(f"Angle of arrival (Y-axis): {np.degrees(theta_y):.2f} deg")
    else:
        print(f"\nWarning: c·τ/d = {config.speed_of_sound * tau_avg_y / config.spacing:.3f} exceeds [-1, 1]")
        print("Cannot calculate angle (may indicate noise or incorrect parameters)")

    print("\n" + "="*60)

    # Plot time vs. phase for all microphones
    fig = plt.figure(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, num_mics))

    for i in range(num_mics):
        plt.plot(t * 1000, phases[i], label=f'Mic {i+1}', color=colors[i], alpha=0.7, linewidth=1)

    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Phase (radians)', fontsize=12)
    plt.title(f'Instantaneous Phase vs. Time - {config.N}x{config.N} Microphone Array (f = {config.frequency} Hz)', fontsize=14)
    plt.legend(ncol=4, fontsize=8, loc='upper left')
    plt.grid(True, alpha=0.3)

    # Save to Pictures folder
    plt.savefig('Pictures/phase_vs_time_array.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to Pictures/phase_vs_time_array.png ({num_mics} microphones)")

    # Add ESC key handler to close plot
    def on_key(event):
        if event.key == 'escape':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
