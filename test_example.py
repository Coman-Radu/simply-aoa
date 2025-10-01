import numpy as np
import matplotlib.pyplot as plt
import json
from mic_array_plot import (
    MicArrayConfig,
    generate_mic_positions,
    apply_steering_vector
)


def load_test_config(filename='config.json'):
    """Load test configuration from JSON"""
    with open(filename, 'r') as f:
        config = json.load(f)
    return config


def simulate_source_signal(mic_positions, config, source_config, sim_config):
    """
    Simulate received signals at each microphone from a source

    Parameters:
    -----------
    mic_positions : ndarray
        Microphone positions
    config : MicArrayConfig
        Mic array configuration
    source_config : dict
        Source configuration (azimuth, elevation, radius, etc.)
    sim_config : dict
        Simulation configuration
    """
    # Extract parameters
    azimuth_rad = np.radians(source_config['azimuth_deg'])
    elevation_rad = np.radians(source_config['elevation_deg'])
    radius = source_config['radius']
    source_freq = source_config['frequency']
    amplitude = source_config['amplitude']

    sample_rate = sim_config['sample_rate']
    duration = sim_config['duration_sec']
    add_noise = sim_config['add_noise']
    noise_level = sim_config['noise_level']

    # Generate time vector
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    # Generate source signal (sine wave)
    source_signal = amplitude * np.sin(2 * np.pi * source_freq * t)

    # Calculate source position in Cartesian coordinates (z is up)
    # Source is at distance 'radius' from origin
    source_x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
    source_y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
    source_z = radius * np.sin(elevation_rad)

    print(f"Source position: ({source_x:.2f}, {source_y:.2f}, {source_z:.2f}) m")
    print(f"Source angles: Azimuth={source_config['azimuth_deg']}°, Elevation={source_config['elevation_deg']}°")

    # Calculate time delays for each microphone
    num_mics = len(mic_positions)
    received_signals = np.zeros((num_mics, num_samples))

    # For each microphone, calculate distance to source and time delay
    time_delays = np.zeros(num_mics)

    for i in range(num_mics):
        # Mic position (assume mics are at z=0 plane)
        mic_x, mic_y = mic_positions[i]
        mic_z = 0.0

        # Distance from source to this mic
        distance = np.sqrt((source_x - mic_x)**2 + (source_y - mic_y)**2 + (source_z - mic_z)**2)

        # Time delay = distance / speed_of_sound
        time_delays[i] = distance / config.speed_of_sound

        # Apply time delay to signal
        delay_samples = int(time_delays[i] * sample_rate)

        if delay_samples < num_samples:
            received_signals[i, delay_samples:] = source_signal[:num_samples - delay_samples]

    # Normalize time delays relative to minimum (first arriving signal)
    min_delay = np.min(time_delays)
    relative_delays = time_delays - min_delay

    # Add noise if enabled
    if add_noise:
        noise = np.random.normal(0, noise_level, received_signals.shape)
        received_signals += noise

    return received_signals, relative_delays, t


def beamform_scan(received_signals, mic_positions, config, angle_range=90, angle_step=5):
    """
    Scan azimuth and elevation angles using steering vectors

    Returns:
    --------
    results : list of tuples (azimuth, elevation, power_db)
    """
    test_angles = np.arange(0, angle_range + 1, angle_step)
    results = []

    for az in test_angles:
        for el in test_angles:
            az_rad = np.radians(az)
            el_rad = np.radians(el)

            # Apply azimuth steering (x-axis)
            steered_az = apply_steering_vector(received_signals, mic_positions, config, az_rad, axis='x')

            # Apply elevation steering (y-axis)
            steered = apply_steering_vector(steered_az, mic_positions, config, el_rad, axis='y')

            # Calculate output power
            output = np.sum(steered, axis=0)
            power = np.mean(np.abs(output)**2)
            power_db = 10 * np.log10(power + 1e-10)

            results.append((az, el, power_db))

    return results


def main():
    # Load configuration
    config_data = load_test_config('config.json')

    # Create mic array config
    mic_config = MicArrayConfig(**config_data['mic_array_config'])
    source_config = config_data['source_config']
    sim_config = config_data['simulation_config']

    # Generate microphone positions
    mic_positions = generate_mic_positions(mic_config)

    print("="*70)
    print("MICROPHONE ARRAY SETUP")
    print("="*70)
    print(f"Array size: {mic_config.N}x{mic_config.N}")
    print(f"Frequency: {mic_config.frequency} Hz")
    print(f"Wavelength: {mic_config.wavelength*1000:.2f} mm")
    print(f"Spacing (λ/2): {mic_config.spacing*1000:.2f} mm")
    print(f"Number of mics: {len(mic_positions)}")

    print("\n" + "="*70)
    print("SOURCE SIMULATION")
    print("="*70)

    # Simulate source signal
    received_signals, time_delays, t = simulate_source_signal(
        mic_positions, mic_config, source_config, sim_config
    )

    print(f"\nRelative time delays (ms):")
    delay_matrix = time_delays.reshape(mic_config.N, mic_config.N) * 1000
    print(delay_matrix)

    print("\n" + "="*70)
    print("BEAMFORMING SCAN")
    print("="*70)

    # Perform beamforming scan
    results = beamform_scan(received_signals, mic_positions, mic_config)

    # Find maximum
    max_power = -np.inf
    best_azimuth = 0
    best_elevation = 0

    for az, el, power in results:
        if power > max_power:
            max_power = power
            best_azimuth = az
            best_elevation = el

    print(f"\nTrue source direction:")
    print(f"  Azimuth: {source_config['azimuth_deg']}°")
    print(f"  Elevation: {source_config['elevation_deg']}°")

    print(f"\nEstimated source direction:")
    print(f"  Azimuth: {best_azimuth}°")
    print(f"  Elevation: {best_elevation}°")

    print(f"\nEstimation error:")
    print(f"  Azimuth error: {abs(best_azimuth - source_config['azimuth_deg']):.1f}°")
    print(f"  Elevation error: {abs(best_elevation - source_config['elevation_deg']):.1f}°")

    # Plot results
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)

    # Create heatmap
    angle_step = 5
    grid_size = 90 // angle_step + 1
    power_grid = np.array([r[2] for r in results]).reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Beamforming heatmap
    ax1 = axes[0]
    im = ax1.imshow(power_grid, extent=[0, 90, 0, 90], origin='lower',
                    cmap='hot', aspect='auto')
    ax1.set_xlabel('Elevation (degrees)', fontsize=12)
    ax1.set_ylabel('Azimuth (degrees)', fontsize=12)
    ax1.set_title('Beamformer Power vs Direction', fontsize=14, fontweight='bold')

    # Mark true and estimated positions
    ax1.plot(source_config['elevation_deg'], source_config['azimuth_deg'],
            'b*', markersize=20, label='True Source', markeredgecolor='white', markeredgewidth=2)
    ax1.plot(best_elevation, best_azimuth,
            'g^', markersize=15, label='Estimated', markeredgecolor='white', markeredgewidth=2)
    ax1.legend(fontsize=10)
    plt.colorbar(im, ax=ax1, label='Power (dB)')

    # Plot 2: Time delays
    ax2 = axes[1]
    im2 = ax2.imshow(delay_matrix, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Column', fontsize=12)
    ax2.set_ylabel('Row', fontsize=12)
    ax2.set_title('Relative Time Delays (ms)', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Time Delay (ms)')

    # Add text annotations for time delays
    for i in range(mic_config.N):
        for j in range(mic_config.N):
            text = ax2.text(j, i, f'{delay_matrix[i, j]:.2f}',
                          ha="center", va="center", color="w", fontsize=8)

    plt.tight_layout()
    plt.savefig('Pictures/test_example_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to Pictures/test_example_results.png")

    plt.show()


if __name__ == "__main__":
    main()