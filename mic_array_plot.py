import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json

@dataclass
class MicArrayConfig:
    """Configuration for microphone array"""
    frequency: float = 100.0  # Hz
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


@dataclass
class SoundPropagationConfig:
    """Configuration for sound propagation simulation in air"""
    source_position: tuple = (0.0, 0.0, 1.0)  # (x, y, z) in meters
    source_amplitude: float = 1.0  # Pa (Pascals)
    frequency: float = 1000.0  # Hz
    speed_of_sound: float = 343.0  # m/s in air at 20°C
    temperature: float = 20.0  # °C
    humidity: float = 50.0  # %
    atmospheric_pressure: float = 101325.0  # Pa
    attenuation_enabled: bool = True  # Enable atmospheric attenuation

    @property
    def wavelength(self):
        """Calculate wavelength"""
        return self.speed_of_sound / self.frequency

    @property
    def wavenumber(self):
        """Calculate wavenumber k = 2π/λ"""
        return 2 * np.pi / self.wavelength

    def adjust_speed_of_sound(self):
        """Adjust speed of sound based on temperature"""
        # Speed of sound in air: c = 331.3 + 0.606 * T (where T is in °C)
        return 331.3 + 0.606 * self.temperature

    def attenuation_coefficient(self):
        """Calculate atmospheric attenuation coefficient (simplified model)"""
        # Simplified attenuation in dB/m (frequency dependent)
        # α ≈ f² (very simplified model)
        if self.attenuation_enabled:
            # More accurate models would include humidity and temperature effects
            return 1e-11 * (self.frequency ** 2)  # Nepers/m
        return 0.0


def load_config_from_json(filename='config.json'):
    """Load configuration from JSON file"""
    with open(filename, 'r') as f:
        config_data = json.load(f)

    mic_array_config = MicArrayConfig(**config_data['mic_array_config'])

    # Convert source_position list to tuple
    sound_prop_data = config_data['sound_propagation_config'].copy()
    sound_prop_data['source_position'] = tuple(sound_prop_data['source_position'])
    sound_propagation_config = SoundPropagationConfig(**sound_prop_data)

    return mic_array_config, sound_propagation_config


def save_config_to_json(mic_array_config, sound_propagation_config, filename='config.json'):
    """Save configuration to JSON file"""
    config_data = {
        'mic_array_config': {
            'frequency': mic_array_config.frequency,
            'speed_of_sound': mic_array_config.speed_of_sound,
            'N': mic_array_config.N
        },
        'sound_propagation_config': {
            'source_position': list(sound_propagation_config.source_position),
            'source_amplitude': sound_propagation_config.source_amplitude,
            'frequency': sound_propagation_config.frequency,
            'speed_of_sound': sound_propagation_config.speed_of_sound,
            'temperature': sound_propagation_config.temperature,
            'humidity': sound_propagation_config.humidity,
            'atmospheric_pressure': sound_propagation_config.atmospheric_pressure,
            'attenuation_enabled': sound_propagation_config.attenuation_enabled
        }
    }

    with open(filename, 'w') as f:
        json.dump(config_data, f, indent=2)


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

    Rearranged formula: theta = arcsin((2 * c * delta_t) / lambda)
    From: delta_t = (lambda * sin(theta)) / (2 * c)

    Parameters:
    -----------
    delta_t : float
        Time delay in seconds between microphones
    wavelength : float
        Wavelength of the signal (lambda) in meters
    speed_of_sound : float
        Speed of sound (c) in m/s

    Returns:
    --------
    theta : float
        Angle of arrival in radians
    """
    theta = np.arcsin((2 * speed_of_sound * delta_t) / wavelength)
    return theta


def calculate_aoa_per_mic(mic_positions, config: MicArrayConfig, delta_t_unit, axis='x'):
    """
    Calculate angle of arrival at each microphone based on time delays
    relative to the center line (t=0).

    For a 4x4 array, row 1 (leftmost to rightmost) has time delays:
    - Mic 1: δt = -3Δt/2
    - Mic 2: δt = -Δt/2
    - Mic 3: δt = +Δt/2
    - Mic 4: δt = +3Δt/2

    In general, δt = k * Δt_unit, where k is the distance index from center
    in units of spacing.

    Parameters:
    -----------
    mic_positions : ndarray
        Microphone positions (shape: [num_mics, 2])
    config : MicArrayConfig
        Configuration object
    delta_t_unit : float
        Base time delay between adjacent mics (seconds)
    axis : str
        'x' for row-wise (azimuth), 'y' for column-wise (elevation)

    Returns:
    --------
    aoa_angles : ndarray
        Angle of arrival in radians for each microphone
    delta_t : ndarray
        Time delay for each microphone in seconds
    """
    d = config.spacing

    # Calculate k (index from centerline) for each microphone
    # k = distance from origin / spacing
    k_x = mic_positions[:, 0] / d
    k_y = mic_positions[:, 1] / d

    # Select appropriate k based on axis
    k = k_x if axis == 'x' else k_y

    # Calculate time delay for each mic: delta_t = k * delta_t_unit
    delta_t = k * delta_t_unit

    # Calculate AoA for each mic using the trig formula
    aoa_angles = apply_trig_angle(delta_t, config.wavelength, config.speed_of_sound)

    return aoa_angles, delta_t


# def apply_steering_vector(x_n, mic_positions, config: MicArrayConfig, theta, axis='x'):
#     """
#     Apply steering vector to microphone signals

#     Parameters:
#     -----------
#     x_n : array_like
#         Input signal for each microphone (shape: [num_mics] or [num_mics, num_samples])
#     mic_positions : ndarray
#         Microphone positions (shape: [num_mics, 2])
#     config : MicArrayConfig
#         Configuration object
#     theta : float
#         Angle of arrival in radians
#     axis : str
#         'x' for azimuth (row-wise), 'y' for elevation (column-wise)

#     Returns:
#     --------
#     y : ndarray
#         Steered signal: x[n] * exp(-2j*pi*d*k*sin(theta))
#         where k is distance from centerline in units of spacing
#     """
#     d = config.spacing

#     # Calculate k (index from centerline) for each microphone
#     # k = distance from origin / spacing
#     k_x = mic_positions[:, 0] / d
#     k_y = mic_positions[:, 1] / d

#     # Calculate steering phase shift for each mic
#     if axis == 'x':
#         # Azimuth: steering along x-axis (horizontal/row-wise)
#         phase_shift = np.exp(-2j * np.pi * d * k_x * np.sin(theta))
#     elif axis == 'y':
#         # Elevation: steering along y-axis (vertical/column-wise)
#         phase_shift = np.exp(-2j * np.pi * d * k_y * np.sin(theta))
#     else:
#         raise ValueError("axis must be 'x' or 'y'")

#     # Apply steering vector
#     if x_n.ndim == 1:
#         y = x_n * phase_shift
#     else:
#         y = x_n * phase_shift[:, np.newaxis]

#     return y


# def apply_2d_steering_vector(x_n, mic_positions, config: MicArrayConfig, azimuth, elevation):
#     """
#     Apply 2D steering vector (both azimuth and elevation)

#     Parameters:
#     -----------
#     x_n : array_like
#         Input signal for each microphone
#     mic_positions : ndarray
#         Microphone positions (shape: [num_mics, 2])
#     config : MicArrayConfig
#         Configuration object
#     azimuth : float
#         Azimuth angle (horizontal/x-axis) in radians
#     elevation : float
#         Elevation angle (vertical/y-axis) in radians

#     Returns:
#     --------
#     y : ndarray
#         Steered signal with both azimuth and elevation applied
#     """
#     d = config.spacing

#     # Calculate k (index from centerline) for each microphone
#     k_x = mic_positions[:, 0] / d
#     k_y = mic_positions[:, 1] / d

#     # Combined phase shift for 2D steering
#     phase_shift = np.exp(-2j * np.pi * d * (k_x * np.sin(azimuth) + k_y * np.sin(elevation)))

#     # Apply steering vector
#     if x_n.ndim == 1:
#         y = x_n * phase_shift
#     else:
#         y = x_n * phase_shift[:, np.newaxis]

#     return y


# def plot_phase_shift_pattern(mic_positions, config: MicArrayConfig, mic_index=None):
#     """
#     Plot phase shift pattern as a function of angle for a given microphone

#     Parameters:
#     -----------
#     mic_positions : ndarray
#         Microphone positions
#     config : MicArrayConfig
#         Configuration object
#     mic_index : int or None
#         Index of microphone to plot. If None, plots all edge microphones
#     """
#     # Range of angles to test (0 to 360 degrees)
#     angles = np.linspace(0, 2*np.pi, 360)

#     if mic_index is None:
#         # Plot for several microphones
#         mic_indices = [0, 3, 12, 15]  # corners of 4x4 array
#     else:
#         mic_indices = [mic_index]

#     fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

#     for idx in mic_indices:
#         phases = []
#         for theta in angles:
#             x_n = np.zeros(len(mic_positions), dtype=complex)
#             x_n[idx] = 1.0
#             y = apply_steering_vector(x_n, mic_positions, config, theta)
#             phases.append(np.angle(y[idx]))

#         # Get mic position
#         x_pos, y_pos = mic_positions[idx]
#         k_val = x_pos / config.spacing

#         ax.plot(angles, phases, label=f'Mic {idx+1} (k={k_val:.1f})', linewidth=2)

#     ax.set_title(f'Phase Shift vs Angle of Arrival\n{config.rows}×{config.cols} array, f={config.frequency}Hz',
#                  fontsize=14, fontweight='bold', pad=20)
#     ax.set_theta_zero_location('E')
#     ax.set_theta_direction(1)
#     ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
#     ax.grid(True)

#     return fig, ax


def plot_mic_array(config: MicArrayConfig):
    """Plot the microphone array"""
    mic_positions = generate_mic_positions(config)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot microphones
    ax.scatter(mic_positions[:, 0], mic_positions[:, 1],
               s=200, c='blue', marker='o', edgecolors='black', linewidth=2,
               label='Microphones')

    # Add microphone labels
    for i, (x, y) in enumerate(mic_positions):
        ax.annotate(f'{i+1}', (x, y), ha='center', va='center',
                   fontsize=8, color='white', fontweight='bold')

    # Add grid lines to show spacing
    for x in np.unique(mic_positions[:, 0]):
        ax.axvline(x, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    for y in np.unique(mic_positions[:, 1]):
        ax.axhline(y, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

    # Add centerlines at origin (array center)
    ax.axvline(0, color='black', linestyle='-', alpha=0.7, linewidth=1.5, label='Center line')
    ax.axhline(0, color='black', linestyle='-', alpha=0.7, linewidth=1.5)

    # Set labels and title
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Microphone Array Layout\n' +
                f'{config.rows}×{config.cols} array, ' +
                f'f={config.frequency}Hz, λ/2={config.spacing*1000:.2f}mm',
                fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)

    # Add info text
    info_text = (f'Frequency: {config.frequency} Hz\n'
                f'Speed of sound: {config.speed_of_sound} m/s\n'
                f'Wavelength (λ): {config.wavelength*1000:.2f} mm\n'
                f'Spacing (λ/2): {config.spacing*1000:.2f} mm\n'
                f'Total mics: {config.rows * config.cols}')

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig, ax, mic_positions


if __name__ == "__main__":
    # Example configurations

    # Configuration 1: 4x4 array at 3.5kHz
    config1 = MicArrayConfig(frequency=3500, N=4)
    fig1, ax1, pos1 = plot_mic_array(config1)
    plt.savefig('Pictures/mic_array_4x4_3500Hz.png', dpi=150, bbox_inches='tight')

    # Configuration 2: 2x2 array at 3.5kHz
    config2 = MicArrayConfig(frequency=3500, N=2)
    fig2, ax2, pos2 = plot_mic_array(config2)
    plt.savefig('Pictures/mic_array_2x2_3500Hz.png', dpi=150, bbox_inches='tight')

    # Configuration 3: 8x8 array at 3.5kHz
    config3 = MicArrayConfig(frequency=3500, N=8)
    fig3, ax3, pos3 = plot_mic_array(config3)
    plt.savefig('Pictures/mic_array_8x8_3500Hz.png', dpi=150, bbox_inches='tight')

    # Configuration 4: 12x12 array at 3.5kHz
    config4 = MicArrayConfig(frequency=3500, N=12)
    fig4, ax4, pos4 = plot_mic_array(config4)
    plt.savefig('Pictures/mic_array_12x12_3500Hz.png', dpi=150, bbox_inches='tight')

    plt.show()

    # Print positions for first config
    print("Microphone positions (config1):")
    print(pos1)

    # Test AoA calculation per microphone
    print("\n" + "="*60)
    print("TESTING ANGLE OF ARRIVAL PER MICROPHONE")
    print("="*60)

    # Test parameter: hypothetical time delay per spacing unit
    # NOTE: This is NOT simulating real delays from a source
    # It's testing the calculate_aoa_per_mic() function by asking:
    # "If we had a 10μs delay between adjacent mics, what angle does that correspond to?"
    # Real source delays are calculated later in the source signal simulation
    delta_t_unit = 10e-6  # seconds (time delay between adjacent mics)

    # Calculate AoA for x-axis (row-wise / azimuth)
    aoa_x, delta_t_x = calculate_aoa_per_mic(pos1, config1, delta_t_unit, axis='x')

    # Calculate AoA for y-axis (column-wise / elevation)
    aoa_y, delta_t_y = calculate_aoa_per_mic(pos1, config1, delta_t_unit, axis='y')

    print(f"\nBase time delay (delta_t): {delta_t_unit*1e6:.2f} us")
    print(f"Wavelength: {config1.wavelength:.3f} m")
    print(f"Spacing (lambda/2): {config1.spacing:.3f} m")

    print("\n" + "-"*70)
    print("AZIMUTH (X-axis / Row-wise) Analysis")
    print("-"*70)

    # Reshape to grid for display
    delta_t_x_grid = delta_t_x.reshape(config1.rows, config1.cols)
    aoa_x_grid = aoa_x.reshape(config1.rows, config1.cols)

    print("\nTime delays (us) - each row should have same pattern:")
    print(delta_t_x_grid * 1e6)

    print("\nAngles of Arrival (degrees) - each row should have same pattern:")
    print(np.degrees(aoa_x_grid))

    print("\n" + "-"*70)
    print("ELEVATION (Y-axis / Column-wise) Analysis")
    print("-"*70)

    delta_t_y_grid = delta_t_y.reshape(config1.rows, config1.cols)
    aoa_y_grid = aoa_y.reshape(config1.rows, config1.cols)

    print("\nTime delays (us) - each column should have same pattern:")
    print(delta_t_y_grid * 1e6)

    print("\nAngles of Arrival (degrees) - each column should have same pattern:")
    print(np.degrees(aoa_y_grid))

    # Test with simulated source signal
    print("\n" + "="*60)
    print("TESTING WITH SIMULATED SOURCE SIGNAL")
    print("="*60)

    # Define source position (x, y, z) in meters
    # Move source much farther away for far-field approximation (planar wavefront)
    source_pos = np.array([50.0, 30.0, 100.0])  # 50m right, 30m forward, 100m up

    print(f"\nSource position: x={source_pos[0]}m, y={source_pos[1]}m, z={source_pos[2]}m")

    # Calculate distance from source to each microphone
    # Microphones are at z=0, so add z coordinate
    mic_positions_3d = np.column_stack([pos1, np.zeros(len(pos1))])

    # Calculate distances and time delays
    distances = np.linalg.norm(mic_positions_3d - source_pos, axis=1)

    # Time delays relative to closest mic
    min_distance = distances.min()
    relative_distances = distances - min_distance
    time_delays = relative_distances / config1.speed_of_sound

    print(f"\nDistances from source to each mic (m):")
    print(distances.reshape(config1.rows, config1.cols))

    print(f"\nTime delays (us):")
    print((time_delays * 1e6).reshape(config1.rows, config1.cols))

    # Calculate theoretical angles from source geometry
    # Source direction angle: direction to the source from array
    theoretical_angle_x = np.arctan2(source_pos[0], source_pos[2])
    theoretical_angle_y = np.arctan2(source_pos[1], source_pos[2])

    print(f"\nTheoretical source direction angle from array normal:")
    print(f"  X-axis (azimuth): {np.degrees(theoretical_angle_x):.2f} degrees")
    print(f"  Y-axis (elevation): {np.degrees(theoretical_angle_y):.2f} degrees")
    print(f"  (Source at x={source_pos[0]}m, y={source_pos[1]}m, z={source_pos[2]}m)")

    # Calculate AoA using centerline as t=0 reference
    # Use the center microphones as reference (indices 5 and 6 for 4x4 array)
    # Center of array is between indices 5,6,9,10 - use average
    center_indices = [5, 6, 9, 10]
    center_time_delay = np.mean(time_delays[center_indices])

    # Normalize time delays relative to center (centerline = t=0)
    time_delays_centered = time_delays - center_time_delay

    print(f"\nTime delays relative to centerline (us):")
    print((time_delays_centered * 1e6).reshape(config1.rows, config1.cols))

    # For x-axis: calculate delta_t per spacing unit using any row
    # Using first row - calculate time delay difference between adjacent mics
    # Negate the time delay to convert to source direction (not wavefront direction)
    row_0_indices = [0, 1, 2, 3]
    row_0_centered_delays = time_delays_centered[row_0_indices]
    # Average the time delay differences between adjacent mics
    # Negative sign: convert wavefront arrival to source direction
    delta_t_x_measured = -np.mean(np.diff(row_0_centered_delays))

    # For y-axis: calculate delta_t per spacing unit using any column
    col_0_indices = [0, 4, 8, 12]
    col_0_centered_delays = time_delays_centered[col_0_indices]
    # Negative sign: convert wavefront arrival to source direction
    delta_t_y_measured = -np.mean(np.diff(col_0_centered_delays))

    aoa_x_calculated = apply_trig_angle(delta_t_x_measured, config1.wavelength, config1.speed_of_sound)
    aoa_y_calculated = apply_trig_angle(delta_t_y_measured, config1.wavelength, config1.speed_of_sound)

    print(f"\nCalculated AoA from time delays (centerline reference):")
    print(f"  X-axis: {np.degrees(aoa_x_calculated):.2f} degrees (delta_t = {delta_t_x_measured*1e6:.2f} us per spacing)")
    print(f"  Y-axis: {np.degrees(aoa_y_calculated):.2f} degrees (delta_t = {delta_t_y_measured*1e6:.2f} us per spacing)")

    # Convert to 0-360 degree system
    aoa_x_360 = np.degrees(aoa_x_calculated) % 360
    aoa_y_360 = np.degrees(aoa_y_calculated) % 360

    print(f"\nAoA in 0-360 degree system:")
    print(f"  X-axis: {aoa_x_360:.2f} degrees")
    print(f"  Y-axis: {aoa_y_360:.2f} degrees")

    # 3D Direction Vector from 2D angles (Triangulation)
    print("\n" + "-"*70)
    print("3D DIRECTION VECTOR FROM 2D ANGLES")
    print("-"*70)

    # We have two angles:
    # - aoa_x_calculated: angle from z-axis in x-z plane (azimuth)
    # - aoa_y_calculated: angle from z-axis in y-z plane (elevation)

    # Build 3D unit vector from the two 2D angle measurements
    # The source direction can be constructed as:
    # x-component: projection onto x-axis from x-z plane angle
    # y-component: projection onto y-axis from y-z plane angle
    # z-component: derived to make unit vector

    dir_x = np.sin(aoa_x_calculated)
    dir_y = np.sin(aoa_y_calculated)

    # For far-field planar wavefront assumption:
    # sin²(θ_x) + sin²(θ_y) + cos²(θ_combined) = 1
    # Solve for z-component
    sin_sum_sq = dir_x**2 + dir_y**2

    if sin_sum_sq <= 1.0:
        dir_z = np.sqrt(1.0 - sin_sum_sq)
    else:
        # Edge case: angles inconsistent (shouldn't happen in far-field)
        # Normalize to maintain unit vector
        norm = np.sqrt(sin_sum_sq)
        dir_x /= norm
        dir_y /= norm
        dir_z = 0.0
        print("Warning: angle sum exceeds physical limit, normalizing...")

    calculated_direction_vector = np.array([dir_x, dir_y, dir_z])

    # Calculate theoretical 3D direction for comparison
    theoretical_direction_vector = source_pos / np.linalg.norm(source_pos)

    print(f"\nCalculated 3D unit vector:")
    print(f"  Direction: [{calculated_direction_vector[0]:.4f}, {calculated_direction_vector[1]:.4f}, {calculated_direction_vector[2]:.4f}]")
    print(f"  Magnitude: {np.linalg.norm(calculated_direction_vector):.4f}")

    print(f"\nTheoretical 3D unit vector:")
    print(f"  Direction: [{theoretical_direction_vector[0]:.4f}, {theoretical_direction_vector[1]:.4f}, {theoretical_direction_vector[2]:.4f}]")
    print(f"  Magnitude: {np.linalg.norm(theoretical_direction_vector):.4f}")

    # Calculate 3D angular error (angle between two vectors)
    dot_product = np.dot(calculated_direction_vector, theoretical_direction_vector)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Handle numerical precision
    angular_error_3d = np.arccos(dot_product)

    print(f"\n3D Angular Error: {np.degrees(angular_error_3d):.2f} degrees")
    print(f"  (Angle between calculated and theoretical direction vectors)")

    # Convert to spherical coordinates for intuition
    # Azimuth φ (in x-y plane from +x axis)
    azimuth_3d_calc = np.arctan2(calculated_direction_vector[1], calculated_direction_vector[0])
    azimuth_3d_theo = np.arctan2(theoretical_direction_vector[1], theoretical_direction_vector[0])

    # Polar angle θ (from +z axis)
    polar_3d_calc = np.arccos(calculated_direction_vector[2])
    polar_3d_theo = np.arccos(theoretical_direction_vector[2])

    print(f"\nSpherical coordinates (for reference):")
    print(f"  Calculated - Azimuth: {np.degrees(azimuth_3d_calc):.2f}°, Polar: {np.degrees(polar_3d_calc):.2f}°")
    print(f"  Theoretical - Azimuth: {np.degrees(azimuth_3d_theo):.2f}°, Polar: {np.degrees(polar_3d_theo):.2f}°")

    print(f"\nTheoretical vs Calculated (2D projections):")
    print(f"  X-axis: {np.degrees(theoretical_angle_x):.2f} deg (theory) vs {np.degrees(aoa_x_calculated):.2f} deg (calc)")
    print(f"  Y-axis: {np.degrees(theoretical_angle_y):.2f} deg (theory) vs {np.degrees(aoa_y_calculated):.2f} deg (calc)")

    print(f"\nError:")
    print(f"  X-axis: {np.degrees(aoa_x_calculated - theoretical_angle_x):.2f} degrees")
    print(f"  Y-axis: {np.degrees(aoa_y_calculated - theoretical_angle_y):.2f} degrees")

    # Plot unit vectors from each mic toward source (3D)
    print("\nGenerating visualization of source direction vectors...")
    fig_vectors = plt.figure(figsize=(16, 8))

    # Create 2 subplots: 3D array closeup and 3D full view
    ax_array = fig_vectors.add_subplot(121, projection='3d')
    ax_3d = fig_vectors.add_subplot(122, projection='3d')

    # LEFT SUBPLOT: Array closeup with unit vectors (3D)
    # Plot microphones on x-y plane (z=0)
    ax_array.scatter(pos1[:, 0], pos1[:, 1], np.zeros(len(pos1)),
                     s=200, c='blue', marker='o', edgecolors='black', linewidth=2,
                     label='Microphones', depthshade=True)

    # Calculate unit vectors from each mic toward source (full 3D)
    for i, mic_pos in enumerate(pos1):
        mic_pos_3d = np.array([mic_pos[0], mic_pos[1], 0])

        # Vector from mic to source (full 3D)
        direction_vector = source_pos - mic_pos_3d

        # Normalize to unit vector
        direction_magnitude = np.linalg.norm(direction_vector)
        if direction_magnitude > 0:
            unit_vector = direction_vector / direction_magnitude

            # Scale for visualization (arrow length relative to array size)
            arrow_scale = np.ptp(pos1[:, 0]) * 0.8  # 80% of array width

            # Draw 3D arrow from mic position
            ax_array.quiver(mic_pos[0], mic_pos[1], 0,
                           unit_vector[0] * arrow_scale,
                           unit_vector[1] * arrow_scale,
                           unit_vector[2] * arrow_scale,
                           color='red', alpha=0.7, arrow_length_ratio=0.2, linewidth=2)

    # Add microphone labels
    for i, (x, y) in enumerate(pos1):
        ax_array.text(x, y, 0, f'{i+1}', ha='center', va='center',
                     fontsize=8, color='white', fontweight='bold', zorder=100,
                     bbox=dict(boxstyle='circle', facecolor='blue', edgecolor='black', linewidth=1.5, pad=0.3))

    # Draw array plane (x-y plane at z=0)
    array_extent = np.ptp(pos1[:, 0]) * 1.3
    xx, yy = np.meshgrid(np.linspace(pos1[:, 0].min() - array_extent*0.2, pos1[:, 0].max() + array_extent*0.2, 10),
                         np.linspace(pos1[:, 1].min() - array_extent*0.2, pos1[:, 1].max() + array_extent*0.2, 10))
    zz = np.zeros_like(xx)
    ax_array.plot_surface(xx, yy, zz, alpha=0.15, color='cyan')

    # Set labels and title
    ax_array.set_xlabel('X (m)', fontsize=10)
    ax_array.set_ylabel('Y (m)', fontsize=10)
    ax_array.set_zlabel('Z (m)', fontsize=10)
    ax_array.set_title(f'3D Array Closeup: Direction Vectors\n' +
                      f'AoA: X={np.degrees(aoa_x_calculated):.2f}°, Y={np.degrees(aoa_y_calculated):.2f}°',
                      fontsize=11, fontweight='bold')

    ax_array.legend(fontsize=9, loc='upper left')

    # Adjust axis limits for closeup view
    array_margin = np.ptp(pos1[:, 0]) * 1.3
    ax_array.set_xlim(pos1[:, 0].min() - array_margin*0.3, pos1[:, 0].max() + array_margin*0.3)
    ax_array.set_ylim(pos1[:, 1].min() - array_margin*0.3, pos1[:, 1].max() + array_margin*0.3)
    ax_array.set_zlim(0, array_margin)

    # Set viewing angle
    ax_array.view_init(elev=25, azim=45)

    # RIGHT SUBPLOT: 3D full view
    # Plot microphones in 3D (at z=0)
    ax_3d.scatter(pos1[:, 0], pos1[:, 1], np.zeros(len(pos1)),
                  s=100, c='blue', marker='o', edgecolors='black', linewidth=1.5,
                  label='Microphone Array', depthshade=True)

    # Plot source position
    ax_3d.scatter(source_pos[0], source_pos[1], source_pos[2],
                  s=400, c='green', marker='*', edgecolors='black', linewidth=2,
                  label=f'Source', depthshade=True)

    # Draw direction vectors from each mic to source
    for i, mic_pos in enumerate(pos1):
        mic_pos_3d = np.array([mic_pos[0], mic_pos[1], 0])
        direction = source_pos - mic_pos_3d
        direction_normalized = direction / np.linalg.norm(direction)

        # Scale arrow for visibility
        arrow_length = np.linalg.norm(source_pos) * 0.15

        ax_3d.quiver(mic_pos[0], mic_pos[1], 0,
                     direction_normalized[0] * arrow_length,
                     direction_normalized[1] * arrow_length,
                     direction_normalized[2] * arrow_length,
                     color='red', alpha=0.6, arrow_length_ratio=0.15, linewidth=1.5)

    # Draw array plane (z=0)
    array_extent = np.ptp(pos1[:, 0]) * 1.5
    xx, yy = np.meshgrid(np.linspace(-array_extent, array_extent, 10),
                         np.linspace(-array_extent, array_extent, 10))
    zz = np.zeros_like(xx)
    ax_3d.plot_surface(xx, yy, zz, alpha=0.1, color='cyan', label='Array Plane')

    # Draw line from array center to source
    array_center = np.mean(pos1, axis=0)
    array_center_3d = np.array([array_center[0], array_center[1], 0])
    ax_3d.plot([array_center_3d[0], source_pos[0]],
               [array_center_3d[1], source_pos[1]],
               [array_center_3d[2], source_pos[2]],
               'g--', linewidth=2, alpha=0.7, label='Center to Source')

    # Labels and title
    ax_3d.set_xlabel('X Position (m)', fontsize=10)
    ax_3d.set_ylabel('Y Position (m)', fontsize=10)
    ax_3d.set_zlabel('Z Position (m)', fontsize=10)
    ax_3d.set_title(f'3D View: Array and Source\n' +
                    f'AoA: X={np.degrees(aoa_x_calculated):.1f}°, Y={np.degrees(aoa_y_calculated):.1f}°',
                    fontsize=12, fontweight='bold')
    ax_3d.legend(fontsize=8, loc='upper left')

    # Set equal aspect ratio for 3D
    max_range = max(abs(source_pos[0]), abs(source_pos[1]), source_pos[2])
    ax_3d.set_xlim([-max_range*0.6, max_range*0.6])
    ax_3d.set_ylim([-max_range*0.6, max_range*0.6])
    ax_3d.set_zlim([0, max_range*1.2])

    # Adjust viewing angle
    ax_3d.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('Pictures/source_direction_vectors.png', dpi=150, bbox_inches='tight')
    print("Saved source direction vectors plot to Pictures/source_direction_vectors.png")

    # Realistic time-domain signal simulation
    print("\n" + "="*60)
    print("REALISTIC TIME-DOMAIN SIGNAL SIMULATION")
    print("="*60)

    # Signal parameters
    signal_freq = config1.frequency  # Use array design frequency
    sample_rate = signal_freq * 20  # Sample at 20x signal frequency (Nyquist + margin)
    duration = 0.01  # 10ms signal duration
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate

    print(f"\nSignal parameters:")
    print(f"  Frequency: {signal_freq} Hz")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration*1000:.1f} ms")
    print(f"  Samples: {num_samples}")

    # Generate source signal (sine wave)
    source_signal = np.sin(2 * np.pi * signal_freq * t)

    # Generate received signals at each microphone
    mic_signals = np.zeros((len(pos1), num_samples))

    for i, mic_pos in enumerate(pos1):
        # Calculate distance from source to mic
        distance = distances[i]

        # Calculate time delay
        delay_time = time_delays[i]
        delay_samples = int(delay_time * sample_rate)

        # Calculate attenuation (1/r spherical spreading)
        attenuation = 1.0 / distance

        # Generate delayed and attenuated signal
        if delay_samples < num_samples:
            # Shift signal by delay_samples
            mic_signals[i, delay_samples:] = source_signal[:num_samples - delay_samples] * attenuation

        # # Add noise (SNR in dB)
        # snr_db = 20  # Signal-to-noise ratio
        # signal_power = np.mean(mic_signals[i]**2)
        # noise_power = signal_power / (10**(snr_db / 10))
        # noise = np.sqrt(noise_power) * np.random.randn(num_samples)
        # mic_signals[i] += noise

    print(f"\nGenerated signals for {len(pos1)} microphones")

    # Cross-correlate between microphones to find time delays
    print("\nCross-correlating signals to measure time delays...")

    # Calculate AoA from cross-correlation (row-wise for X-axis)
    print("\nX-axis (azimuth) - analyzing first row:")
    row_0_indices = [0, 1, 2, 3]

    # Cross-correlate adjacent mics in the row
    delta_t_measured_list = []
    for j in range(len(row_0_indices) - 1):
        mic_a = row_0_indices[j]
        mic_b = row_0_indices[j + 1]

        # Cross-correlate
        correlation = np.correlate(mic_signals[mic_b], mic_signals[mic_a], mode='full')

        # Find peak
        peak_idx = np.argmax(correlation)

        # Convert to time delay (relative to center of correlation)
        delay_samples = peak_idx - (num_samples - 1)
        delay_time = delay_samples / sample_rate

        delta_t_measured_list.append(delay_time)

        print(f"  Mic {mic_a+1} to Mic {mic_b+1}: {delay_time*1e6:.2f} us ({delay_samples} samples)")

    # Average time delay per spacing
    delta_t_x_corr = -np.mean(delta_t_measured_list)  # Negative for source direction
    aoa_x_corr = apply_trig_angle(delta_t_x_corr, config1.wavelength, config1.speed_of_sound)

    print(f"\nAverage delta_t per spacing (X): {delta_t_x_corr*1e6:.2f} us")
    print(f"Calculated AoA (X-axis) from correlation: {np.degrees(aoa_x_corr):.2f} degrees")

    # Calculate AoA from cross-correlation (column-wise for Y-axis)
    print("\nY-axis (elevation) - analyzing first column:")
    col_0_indices = [0, 4, 8, 12]

    delta_t_measured_list_y = []
    for j in range(len(col_0_indices) - 1):
        mic_a = col_0_indices[j]
        mic_b = col_0_indices[j + 1]

        # Cross-correlate
        correlation = np.correlate(mic_signals[mic_b], mic_signals[mic_a], mode='full')

        # Find peak
        peak_idx = np.argmax(correlation)

        # Convert to time delay
        delay_samples = peak_idx - (num_samples - 1)
        delay_time = delay_samples / sample_rate

        delta_t_measured_list_y.append(delay_time)

        print(f"  Mic {mic_a+1} to Mic {mic_b+1}: {delay_time*1e6:.2f} us ({delay_samples} samples)")

    # Average time delay per spacing
    delta_t_y_corr = -np.mean(delta_t_measured_list_y)  # Negative for source direction
    aoa_y_corr = apply_trig_angle(delta_t_y_corr, config1.wavelength, config1.speed_of_sound)

    print(f"\nAverage delta_t per spacing (Y): {delta_t_y_corr*1e6:.2f} us")
    print(f"Calculated AoA (Y-axis) from correlation: {np.degrees(aoa_y_corr):.2f} degrees")

    print(f"\nComparison with theoretical:")
    print(f"  X-axis: Theory={np.degrees(theoretical_angle_x):.2f}°, Correlation={np.degrees(aoa_x_corr):.2f}°, Error={np.degrees(aoa_x_corr - theoretical_angle_x):.2f}°")
    print(f"  Y-axis: Theory={np.degrees(theoretical_angle_y):.2f}°, Correlation={np.degrees(aoa_y_corr):.2f}°, Error={np.degrees(aoa_y_corr - theoretical_angle_y):.2f}°")

    # Test far-field distance criterion
    print("\n" + "="*60)
    print("FAR-FIELD DISTANCE ANALYSIS")
    print("="*60)

    # Calculate array aperture (maximum dimension)
    array_size_x = np.ptp(pos1[:, 0])  # peak-to-peak in x
    array_size_y = np.ptp(pos1[:, 1])  # peak-to-peak in y
    D = max(array_size_x, array_size_y)  # maximum aperture

    print(f"\nArray dimensions:")
    print(f"  X-span: {array_size_x:.3f} m")
    print(f"  Y-span: {array_size_y:.3f} m")
    print(f"  Maximum aperture (D): {D:.3f} m")
    print(f"  Wavelength (lambda): {config1.wavelength:.3f} m")

    # Calculate Fraunhofer distance
    fraunhofer_distance = 2 * D**2 / config1.wavelength

    print(f"\nFar-field criterion (Fraunhofer distance):")
    print(f"  r >= 2*D^2/lambda = {fraunhofer_distance:.2f} m")

    # Test at various distances
    print(f"\nTesting accuracy vs distance:")
    print(f"{'Distance (m)':<15} {'Theory X (deg)':<15} {'Calc X (deg)':<15} {'Error (deg)':<15} {'% Error':<10}")
    print("-" * 80)

    test_distances = [fraunhofer_distance * 0.1, fraunhofer_distance * 0.5,
                      fraunhofer_distance * 1.0, fraunhofer_distance * 2.0,
                      fraunhofer_distance * 5.0, fraunhofer_distance * 10.0]

    for test_dist in test_distances:
        # Place source at same angle but different distance
        # Keep same angles as before
        angle_x_rad = theoretical_angle_x
        angle_y_rad = theoretical_angle_y

        # Convert to source position
        test_source_x = test_dist * np.sin(angle_x_rad) / np.cos(angle_x_rad) * np.cos(angle_x_rad)
        test_source_y = test_dist * np.sin(angle_y_rad) / np.cos(angle_y_rad) * np.cos(angle_y_rad)
        test_source_z = test_dist / np.sqrt(1 + np.tan(angle_x_rad)**2 + np.tan(angle_y_rad)**2)
        test_source_pos = np.array([test_source_x, test_source_y, test_source_z])

        # Calculate distances and time delays
        test_distances_mics = np.linalg.norm(mic_positions_3d - test_source_pos, axis=1)
        test_min_distance = test_distances_mics.min()
        test_relative_distances = test_distances_mics - test_min_distance
        test_time_delays = test_relative_distances / config1.speed_of_sound

        # Calculate AoA
        test_center_time_delay = np.mean(test_time_delays[center_indices])
        test_time_delays_centered = test_time_delays - test_center_time_delay

        test_row_0_centered_delays = test_time_delays_centered[row_0_indices]
        test_delta_t_x = -np.mean(np.diff(test_row_0_centered_delays))

        test_aoa_x = apply_trig_angle(test_delta_t_x, config1.wavelength, config1.speed_of_sound)

        error_deg = np.degrees(test_aoa_x - angle_x_rad)
        error_pct = abs(error_deg / np.degrees(angle_x_rad)) * 100

        print(f"{test_dist:<15.2f} {np.degrees(angle_x_rad):<15.2f} {np.degrees(test_aoa_x):<15.2f} {error_deg:<15.2f} {error_pct:<10.2f}")

    # # Test steering vector
    # print("\n" + "="*60)
    # print("TESTING STEERING VECTOR")
    # print("="*60)

    # # Test with simple signal: all mics receive signal with amplitude 1
    # num_mics = len(pos1)
    # x_n = np.ones(num_mics, dtype=complex)

    # # Test at different angles
    # angles_deg = [0, 30, 45, 90]

    # for angle_deg in angles_deg:
    #     theta = np.radians(angle_deg)

    #     # Test azimuth (x-axis / row-wise)
    #     y_azimuth = apply_steering_vector(x_n, pos1, config1, theta, axis='x')

    #     # Test elevation (y-axis / column-wise)
    #     y_elevation = apply_steering_vector(x_n, pos1, config1, theta, axis='y')

    #     print(f"\n{'='*70}")
    #     print(f"Angle: {angle_deg}° ({theta:.3f} rad)")
    #     print(f"{'='*70}")

    #     # Azimuth phase shifts
    #     phase_azimuth = np.angle(y_azimuth).reshape(config1.rows, config1.cols)
    #     print("\nAZIMUTH (Row-wise) Phase Shift Matrix (degrees):")
    #     print(np.degrees(phase_azimuth))

    #     # Elevation phase shifts
    #     phase_elevation = np.angle(y_elevation).reshape(config1.rows, config1.cols)
    #     print("\nELEVATION (Column-wise) Phase Shift Matrix (degrees):")
    #     print(np.degrees(phase_elevation))

    #     # Combined 2D steering
    #     y_2d = apply_2d_steering_vector(x_n, pos1, config1, theta, theta)
    #     phase_2d = np.angle(y_2d).reshape(config1.rows, config1.cols)
    #     print("\nCOMBINED 2D (Azimuth + Elevation) Phase Shift Matrix (degrees):")
    #     print(np.degrees(phase_2d))

    #     print(f"\nMagnitudes (all should be 1.0):")
    #     magnitude_matrix = np.abs(y_2d).reshape(config1.rows, config1.cols)
    #     print(magnitude_matrix)

    # # Test with time-series data
    # print("\n" + "="*60)
    # print("TESTING WITH TIME-SERIES DATA")
    # print("="*60)

    # num_samples = 100
    # t = np.arange(num_samples) / config1.frequency

    # # Create signal: each mic receives a sine wave
    # signal_freq = 50  # Hz
    # x_n_time = np.zeros((num_mics, num_samples), dtype=complex)
    # for i in range(num_mics):
    #     x_n_time[i, :] = np.sin(2 * np.pi * signal_freq * t)

    # theta = np.radians(30)
    # y_time = apply_steering_vector(x_n_time, pos1, config1, theta)

    # print(f"\nInput shape: {x_n_time.shape}")
    # print(f"Output shape: {y_time.shape}")
    # print(f"First sample, first 4 mics input: {x_n_time[:4, 0]}")
    # print(f"First sample, first 4 mics output: {y_time[:4, 0]}")
    # print(f"Output is complex: {np.iscomplexobj(y_time)}")
    # print(f"Magnitude preserved: {np.allclose(np.abs(y_time), np.abs(x_n_time))}")

    # # Plot phase shift pattern
    # print("\n" + "="*60)
    # print("PLOTTING PHASE SHIFT PATTERN")
    # print("="*60)

    # fig_phase, ax_phase = plot_phase_shift_pattern(pos1, config1)
    # plt.savefig('Pictures/phase_shift_pattern_4x4_100Hz.png', dpi=150, bbox_inches='tight')
    # print("Phase shift pattern saved to Pictures/phase_shift_pattern_4x4_100Hz.png")