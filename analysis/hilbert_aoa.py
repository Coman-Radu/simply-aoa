import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.io import wavfile
from scipy.signal import hilbert


@dataclass
class MicArrayConfig:
    """Configuration for microphone array"""
    frequency: float = 3500.0  # Hz
    speed_of_sound: float = 343.0  # m/s in air at 20°C
    N: int = 4  # NxN array size

    @property
    def rows(self):
        return self.N

    @property
    def cols(self):
        return self.N

    @property
    def wavelength(self):
        return self.speed_of_sound / self.frequency

    @property
    def spacing(self):
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
    theta = arcsin((2 * c * delta_t) / lambda)
    """
    theta = np.arcsin((2 * speed_of_sound * delta_t) / wavelength)
    return theta


def hilbert_phase_tdoa(mic_signals, signal_freq, sample_rate, mic_indices):
    """
    Estimate TDOA using Hilbert transform phase differences

    Parameters:
    -----------
    mic_signals : ndarray
        Signal for each microphone [num_mics, num_samples]
    signal_freq : float
        Signal frequency in Hz
    sample_rate : float
        Sample rate in Hz
    mic_indices : list
        Indices of microphones to analyze (in order)

    Returns:
    --------
    delays : list
        Time delays between adjacent microphones in seconds
    """
    # Compute analytic signals
    analytic_signals = np.array([hilbert(mic_signals[i]) for i in mic_indices])

    # Extract instantaneous phase
    instantaneous_phases = np.angle(analytic_signals)

    # Find peak in envelope for reference
    envelope_ref = np.abs(analytic_signals[0])
    peak_idx = np.argmax(envelope_ref)

    # Calculate phase differences at peak
    delays = []
    for j in range(len(mic_indices) - 1):
        # Phase difference at peak
        phase_diff = instantaneous_phases[j+1, peak_idx] - instantaneous_phases[j, peak_idx]

        # Unwrap phase to [-pi, pi]
        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

        # Convert phase difference to time delay
        # For narrowband signal: dt = phase_diff / (2*pi*f)
        delay_time = phase_diff / (2 * np.pi * signal_freq)

        delays.append(delay_time)

    return delays, peak_idx


def hilbert_envelope_tdoa(mic_signals, sample_rate, mic_indices):
    """
    Estimate TDOA using Hilbert envelope cross-correlation

    Parameters:
    -----------
    mic_signals : ndarray
        Signal for each microphone [num_mics, num_samples]
    sample_rate : float
        Sample rate in Hz
    mic_indices : list
        Indices of microphones to analyze (in order)

    Returns:
    --------
    delays : list
        Time delays between adjacent microphones in seconds
    """
    num_samples = mic_signals.shape[1]

    # Compute analytic signals and extract envelopes
    envelopes = np.array([np.abs(hilbert(mic_signals[i])) for i in mic_indices])

    # Cross-correlate envelopes
    delays = []
    for j in range(len(mic_indices) - 1):
        correlation = np.correlate(envelopes[j+1], envelopes[j], mode='full')
        peak_idx = np.argmax(correlation)

        delay_samples = peak_idx - (num_samples - 1)
        delay_time = delay_samples / sample_rate

        delays.append(delay_time)

    return delays


def simulate_source_signals(mic_positions, source_pos, config, wav_file='drone_monotone_3kHz.wav',
                           duration=None, use_continuous=True):
    """
    Simulate received signals at each microphone from a source

    Parameters:
    -----------
    mic_positions : ndarray
        Microphone positions [num_mics, 2]
    source_pos : ndarray or callable
        Source position [x, y, z] in meters, or function(t) returning position
    config : MicArrayConfig
        Array configuration
    wav_file : str
        Path to wav file to use as source signal
    duration : float or None
        Duration of signal to use in seconds. If None, uses full wav file
    use_continuous : bool
        If True, uses full/longer signal for realistic continuous drone sound

    Returns:
    --------
    mic_signals : ndarray
        Signals at each microphone [num_mics, num_samples]
    sample_rate : int
        Sample rate of signals
    distances : ndarray
        Distances from source to each mic
    time_delays : ndarray
        Time delays at each mic relative to closest mic
    """
    # Load wav file
    sample_rate, wav_data = wavfile.read(wav_file)

    # Convert to float
    if wav_data.dtype == np.int16:
        wav_data = wav_data.astype(np.float32) / 32768.0
    elif wav_data.dtype == np.int32:
        wav_data = wav_data.astype(np.float32) / 2147483648.0

    # If stereo, take first channel
    if wav_data.ndim > 1:
        wav_data = wav_data[:, 0]

    # Determine duration
    if duration is None:
        # Use full wav file
        duration = len(wav_data) / sample_rate
        num_samples = len(wav_data)
        source_signal = wav_data
    elif use_continuous:
        # Use specified duration for continuous simulation (up to wav file length)
        duration = min(len(wav_data) / sample_rate, duration)
        num_samples = int(duration * sample_rate)
        source_signal = wav_data[:num_samples]
    else:
        # Use specified duration
        duration = min(len(wav_data) / sample_rate, duration)
        num_samples = int(duration * sample_rate)
        source_signal = wav_data[:num_samples]

    # Check if source is moving (callable) or static (array)
    if callable(source_pos):
        # Moving source - calculate position over time
        t = np.arange(num_samples) / sample_rate
        source_positions = np.array([source_pos(ti) for ti in t])

        # For moving source, calculate average position for initial distance/delay estimate
        source_pos_static = source_positions[0]
    else:
        # Static source
        source_pos_static = source_pos
        source_positions = None

    # Calculate distances from source to each mic (initial position)
    mic_positions_3d = np.column_stack([mic_positions, np.zeros(len(mic_positions))])
    distances = np.linalg.norm(mic_positions_3d - source_pos_static, axis=1)

    # Time delays relative to closest mic
    min_distance = distances.min()
    relative_distances = distances - min_distance
    time_delays = relative_distances / config.speed_of_sound

    # Generate received signals
    mic_signals = np.zeros((len(mic_positions), num_samples))

    if source_positions is not None:
        # Moving source - calculate Doppler and time-varying delays
        print("  Simulating moving drone source...")
        for i in range(len(mic_positions)):
            mic_pos_3d = mic_positions_3d[i]

            # For each time sample, calculate instantaneous distance
            for t_idx in range(num_samples):
                source_pos_t = source_positions[t_idx]
                distance_t = np.linalg.norm(mic_pos_3d - source_pos_t)

                # Calculate delay from source to mic
                delay_time = distance_t / config.speed_of_sound
                delay_samples = int(delay_time * sample_rate)

                # Calculate attenuation
                attenuation = 1.0 / max(distance_t, 1.0)  # Prevent division by zero

                # Add contribution from this time to the delayed sample
                source_idx = t_idx - delay_samples
                if 0 <= source_idx < num_samples and 0 <= t_idx < num_samples:
                    mic_signals[i, t_idx] += source_signal[source_idx] * attenuation
    else:
        # Static source - simpler calculation
        for i in range(len(mic_positions)):
            distance = distances[i]
            delay_time = time_delays[i]
            delay_samples = int(delay_time * sample_rate)

            # Calculate attenuation (1/r spherical spreading)
            attenuation = 1.0 / distance

            # Generate delayed and attenuated signal
            if delay_samples < num_samples:
                mic_signals[i, delay_samples:] = source_signal[:num_samples - delay_samples] * attenuation

    return mic_signals, sample_rate, distances, time_delays


def plot_mic_array_with_source(mic_positions, config, source_pos, aoa_x, aoa_y):
    """Plot microphone array with source direction vectors"""
    fig = plt.figure(figsize=(16, 8))

    # LEFT: Array closeup with direction vectors
    ax_array = fig.add_subplot(121, projection='3d')

    # Plot microphones
    ax_array.scatter(mic_positions[:, 0], mic_positions[:, 1], np.zeros(len(mic_positions)),
                     s=200, c='blue', marker='o', edgecolors='black', linewidth=2,
                     label='Microphones', depthshade=True)

    # Calculate and plot direction vectors
    for i, mic_pos in enumerate(mic_positions):
        mic_pos_3d = np.array([mic_pos[0], mic_pos[1], 0])
        direction_vector = source_pos - mic_pos_3d

        if np.linalg.norm(direction_vector) > 0:
            unit_vector = direction_vector / np.linalg.norm(direction_vector)
            arrow_scale = np.ptp(mic_positions[:, 0]) * 0.8

            ax_array.quiver(mic_pos[0], mic_pos[1], 0,
                           unit_vector[0] * arrow_scale,
                           unit_vector[1] * arrow_scale,
                           unit_vector[2] * arrow_scale,
                           color='red', alpha=0.7, arrow_length_ratio=0.2, linewidth=2)

    # Add labels
    for i, (x, y) in enumerate(mic_positions):
        ax_array.text(x, y, 0, f'{i+1}', ha='center', va='center',
                     fontsize=8, color='white', fontweight='bold', zorder=100,
                     bbox=dict(boxstyle='circle', facecolor='blue', edgecolor='black', linewidth=1.5, pad=0.3))

    # Draw array plane
    array_extent = np.ptp(mic_positions[:, 0]) * 1.3
    xx, yy = np.meshgrid(np.linspace(mic_positions[:, 0].min() - array_extent*0.2,
                                     mic_positions[:, 0].max() + array_extent*0.2, 10),
                         np.linspace(mic_positions[:, 1].min() - array_extent*0.2,
                                     mic_positions[:, 1].max() + array_extent*0.2, 10))
    zz = np.zeros_like(xx)
    ax_array.plot_surface(xx, yy, zz, alpha=0.15, color='cyan')

    ax_array.set_xlabel('X (m)', fontsize=10)
    ax_array.set_ylabel('Y (m)', fontsize=10)
    ax_array.set_zlabel('Z (m)', fontsize=10)
    ax_array.set_title(f'3D Array: Hilbert Transform AoA\n' +
                      f'X={np.degrees(aoa_x):.2f}°, Y={np.degrees(aoa_y):.2f}°',
                      fontsize=11, fontweight='bold')
    ax_array.legend(fontsize=9, loc='upper left')

    array_margin = np.ptp(mic_positions[:, 0]) * 1.3
    ax_array.set_xlim(mic_positions[:, 0].min() - array_margin*0.3,
                     mic_positions[:, 0].max() + array_margin*0.3)
    ax_array.set_ylim(mic_positions[:, 1].min() - array_margin*0.3,
                     mic_positions[:, 1].max() + array_margin*0.3)
    ax_array.set_zlim(0, array_margin)
    ax_array.view_init(elev=25, azim=45)

    # RIGHT: Full 3D view
    ax_3d = fig.add_subplot(122, projection='3d')

    ax_3d.scatter(mic_positions[:, 0], mic_positions[:, 1], np.zeros(len(mic_positions)),
                  s=100, c='blue', marker='o', edgecolors='black', linewidth=1.5,
                  label='Microphone Array', depthshade=True)

    ax_3d.scatter(source_pos[0], source_pos[1], source_pos[2],
                  s=400, c='green', marker='*', edgecolors='black', linewidth=2,
                  label='Source', depthshade=True)

    # Draw direction vectors
    for i, mic_pos in enumerate(mic_positions):
        mic_pos_3d = np.array([mic_pos[0], mic_pos[1], 0])
        direction = source_pos - mic_pos_3d
        direction_normalized = direction / np.linalg.norm(direction)

        arrow_length = np.linalg.norm(source_pos) * 0.15

        ax_3d.quiver(mic_pos[0], mic_pos[1], 0,
                     direction_normalized[0] * arrow_length,
                     direction_normalized[1] * arrow_length,
                     direction_normalized[2] * arrow_length,
                     color='red', alpha=0.6, arrow_length_ratio=0.15, linewidth=1.5)

    # Draw array plane
    array_extent = np.ptp(mic_positions[:, 0]) * 1.5
    xx, yy = np.meshgrid(np.linspace(-array_extent, array_extent, 10),
                         np.linspace(-array_extent, array_extent, 10))
    zz = np.zeros_like(xx)
    ax_3d.plot_surface(xx, yy, zz, alpha=0.1, color='cyan')

    # Line from array center to source
    array_center = np.mean(mic_positions, axis=0)
    array_center_3d = np.array([array_center[0], array_center[1], 0])
    ax_3d.plot([array_center_3d[0], source_pos[0]],
               [array_center_3d[1], source_pos[1]],
               [array_center_3d[2], source_pos[2]],
               'g--', linewidth=2, alpha=0.7, label='Center to Source')

    ax_3d.set_xlabel('X Position (m)', fontsize=10)
    ax_3d.set_ylabel('Y Position (m)', fontsize=10)
    ax_3d.set_zlabel('Z Position (m)', fontsize=10)
    ax_3d.set_title(f'3D View: Array and Source\n' +
                    f'AoA: X={np.degrees(aoa_x):.1f}°, Y={np.degrees(aoa_y):.1f}°',
                    fontsize=12, fontweight='bold')
    ax_3d.legend(fontsize=8, loc='upper left')

    max_range = max(abs(source_pos[0]), abs(source_pos[1]), source_pos[2])
    ax_3d.set_xlim([-max_range*0.6, max_range*0.6])
    ax_3d.set_ylim([-max_range*0.6, max_range*0.6])
    ax_3d.set_zlim([0, max_range*1.2])
    ax_3d.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Configuration
    config = MicArrayConfig(frequency=3500, N=4)

    # Simulation mode: 'static' or 'moving'
    simulation_mode = 'static'  # Change to 'moving' for drone flight simulation

    # Signal duration for continuous simulation (seconds)
    # Use longer duration for realistic continuous drone sound
    continuous_duration = 1.0  # 1 second of continuous signal

    if simulation_mode == 'static':
        # Static drone hovering at fixed position
        source_pos = np.array([50.0, 30.0, 100.0])  # x, y, z in meters
        use_continuous = True  # Use continuous signal
    else:
        # Moving drone - define trajectory
        # Example: drone flying in a straight line
        initial_pos = np.array([50.0, 30.0, 100.0])
        velocity = np.array([5.0, 2.0, 0.0])  # 5 m/s in x, 2 m/s in y, stationary in z

        def drone_trajectory(t):
            """Drone position as function of time"""
            return initial_pos + velocity * t

        source_pos = drone_trajectory
        use_continuous = True

    print("="*60)
    print("HILBERT TRANSFORM-BASED ANGLE OF ARRIVAL ESTIMATION")
    print("="*60)
    print(f"Simulation mode: {simulation_mode.upper()}")

    # Generate microphone positions
    mic_positions = generate_mic_positions(config)

    print(f"\nArray configuration:")
    print(f"  Frequency: {config.frequency} Hz")
    print(f"  Array size: {config.N}x{config.N}")
    print(f"  Spacing: {config.spacing*1000:.2f} mm")
    print(f"  Wavelength: {config.wavelength*1000:.2f} mm")

    if simulation_mode == 'static':
        print(f"\nSource position: [{source_pos[0]}, {source_pos[1]}, {source_pos[2]}] m")
    else:
        print(f"\nDrone trajectory:")
        print(f"  Initial position: {initial_pos} m")
        print(f"  Velocity: {velocity} m/s")

    # Simulate received signals
    print("\nSimulating microphone signals...")
    mic_signals, sample_rate, distances, time_delays = simulate_source_signals(
        mic_positions, source_pos, config, duration=continuous_duration, use_continuous=use_continuous
    )

    print(f"\nSignal properties:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {mic_signals.shape[1]/sample_rate:.2f} s ({mic_signals.shape[1]/sample_rate*1000:.1f} ms)")
    print(f"  Samples: {mic_signals.shape[1]}")

    # Calculate theoretical angles
    if callable(source_pos):
        # For moving source, use position at middle of signal
        t_mid = (mic_signals.shape[1] / sample_rate) / 2
        source_pos_eval = source_pos(t_mid)
        print(f"\nEvaluating AoA at t={t_mid:.2f}s (middle of signal)")
        print(f"  Drone position at t={t_mid:.2f}s: {source_pos_eval}")
    else:
        source_pos_eval = source_pos

    theoretical_angle_x = np.arctan2(source_pos_eval[0], source_pos_eval[2])
    theoretical_angle_y = np.arctan2(source_pos_eval[1], source_pos_eval[2])

    print(f"\nTheoretical AoA:")
    print(f"  X-axis (azimuth): {np.degrees(theoretical_angle_x):.2f}°")
    print(f"  Y-axis (elevation): {np.degrees(theoretical_angle_y):.2f}°")

    # Hilbert Phase-based TDOA
    print("\n" + "="*60)
    print("METHOD 1: HILBERT PHASE-BASED TDOA")
    print("="*60)

    # X-axis (azimuth) - first row
    row_0_indices = [0, 1, 2, 3]
    phase_delays_x, peak_idx = hilbert_phase_tdoa(mic_signals, config.frequency,
                                                    sample_rate, row_0_indices)

    print(f"\nX-axis (azimuth) - first row:")
    print(f"  Peak envelope at sample {peak_idx} ({peak_idx/sample_rate*1e6:.2f} us)")
    for j, delay in enumerate(phase_delays_x):
        print(f"  Mic {row_0_indices[j]+1} to Mic {row_0_indices[j+1]+1}: {delay*1e6:.2f} us")

    delta_t_x_phase = -np.mean(phase_delays_x)
    aoa_x_phase = apply_trig_angle(delta_t_x_phase, config.wavelength, config.speed_of_sound)

    print(f"\nAverage delta_t: {delta_t_x_phase*1e6:.2f} us")
    print(f"Calculated AoA (X): {np.degrees(aoa_x_phase):.2f}°")
    print(f"Error: {np.degrees(aoa_x_phase - theoretical_angle_x):.2f}°")

    # Y-axis (elevation) - first column
    col_0_indices = [0, 4, 8, 12]
    phase_delays_y, _ = hilbert_phase_tdoa(mic_signals, config.frequency,
                                            sample_rate, col_0_indices)

    print(f"\nY-axis (elevation) - first column:")
    for j, delay in enumerate(phase_delays_y):
        print(f"  Mic {col_0_indices[j]+1} to Mic {col_0_indices[j+1]+1}: {delay*1e6:.2f} us")

    delta_t_y_phase = -np.mean(phase_delays_y)
    aoa_y_phase = apply_trig_angle(delta_t_y_phase, config.wavelength, config.speed_of_sound)

    print(f"\nAverage delta_t: {delta_t_y_phase*1e6:.2f} us")
    print(f"Calculated AoA (Y): {np.degrees(aoa_y_phase):.2f}°")
    print(f"Error: {np.degrees(aoa_y_phase - theoretical_angle_y):.2f}°")

    # Hilbert Envelope-based TDOA
    print("\n" + "="*60)
    print("METHOD 2: HILBERT ENVELOPE CROSS-CORRELATION")
    print("="*60)

    envelope_delays_x = hilbert_envelope_tdoa(mic_signals, sample_rate, row_0_indices)

    print(f"\nX-axis (azimuth):")
    for j, delay in enumerate(envelope_delays_x):
        print(f"  Mic {row_0_indices[j]+1} to Mic {row_0_indices[j+1]+1}: {delay*1e6:.2f} us")

    delta_t_x_env = -np.mean(envelope_delays_x)
    aoa_x_env = apply_trig_angle(delta_t_x_env, config.wavelength, config.speed_of_sound)

    print(f"\nAverage delta_t: {delta_t_x_env*1e6:.2f} us")
    print(f"Calculated AoA (X): {np.degrees(aoa_x_env):.2f}°")
    print(f"Error: {np.degrees(aoa_x_env - theoretical_angle_x):.2f}°")

    envelope_delays_y = hilbert_envelope_tdoa(mic_signals, sample_rate, col_0_indices)

    print(f"\nY-axis (elevation):")
    for j, delay in enumerate(envelope_delays_y):
        print(f"  Mic {col_0_indices[j]+1} to Mic {col_0_indices[j+1]+1}: {delay*1e6:.2f} us")

    delta_t_y_env = -np.mean(envelope_delays_y)
    aoa_y_env = apply_trig_angle(delta_t_y_env, config.wavelength, config.speed_of_sound)

    print(f"\nAverage delta_t: {delta_t_y_env*1e6:.2f} us")
    print(f"Calculated AoA (Y): {np.degrees(aoa_y_env):.2f}°")
    print(f"Error: {np.degrees(aoa_y_env - theoretical_angle_y):.2f}°")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\nX-axis (Azimuth):")
    print(f"  Theoretical:        {np.degrees(theoretical_angle_x):.2f}°")
    print(f"  Phase-based:        {np.degrees(aoa_x_phase):.2f}° (error: {np.degrees(aoa_x_phase - theoretical_angle_x):.2f}°)")
    print(f"  Envelope-based:     {np.degrees(aoa_x_env):.2f}° (error: {np.degrees(aoa_x_env - theoretical_angle_x):.2f}°)")

    print(f"\nY-axis (Elevation):")
    print(f"  Theoretical:        {np.degrees(theoretical_angle_y):.2f}°")
    print(f"  Phase-based:        {np.degrees(aoa_y_phase):.2f}° (error: {np.degrees(aoa_y_phase - theoretical_angle_y):.2f}°)")
    print(f"  Envelope-based:     {np.degrees(aoa_y_env):.2f}° (error: {np.degrees(aoa_y_env - theoretical_angle_y):.2f}°)")

    # Visualization
    print("\nGenerating visualization...")
    fig = plot_mic_array_with_source(mic_positions, config, source_pos_eval, aoa_x_phase, aoa_y_phase)
    plt.savefig('Pictures/hilbert_aoa_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved to Pictures/hilbert_aoa_visualization.png")

    plt.show()
