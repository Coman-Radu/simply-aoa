import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass

@dataclass
class MicArrayConfig:
    """Configuration for microphone array"""
    frequency: float = 3500.0  # Hz
    speed_of_sound: float = 343.0  # m/s in air at 20°C
    N: int = 4  # NxN array size

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
    x_positions = np.arange(config.N) * spacing
    y_positions = np.arange(config.N) * spacing

    # Center the array at origin
    x_positions -= x_positions.mean()
    y_positions -= y_positions.mean()

    # Create mesh grid
    X, Y = np.meshgrid(x_positions, y_positions)

    # Flatten to get individual mic coordinates
    mic_positions = np.column_stack((X.flatten(), Y.flatten()))

    return mic_positions


def apply_trig_angle(delta_t, wavelength, speed_of_sound):
    """Calculate angle of arrival from time delay"""
    return np.arcsin((2 * speed_of_sound * delta_t) / wavelength)


def calculate_aoa_from_source(source_pos, mic_positions, config):
    """Calculate angle of arrival from source position"""
    # Microphones are at z=0
    mic_positions_3d = np.column_stack([mic_positions, np.zeros(len(mic_positions))])

    # Calculate distances and time delays
    distances = np.linalg.norm(mic_positions_3d - source_pos, axis=1)
    min_distance = distances.min()
    relative_distances = distances - min_distance
    time_delays = relative_distances / config.speed_of_sound

    # Calculate center time delay
    center_indices = [5, 6, 9, 10]  # For 4x4 array
    center_time_delay = np.mean(time_delays[center_indices])
    time_delays_centered = time_delays - center_time_delay

    # X-axis: calculate delta_t per spacing unit
    row_0_indices = [0, 1, 2, 3]
    row_0_centered_delays = time_delays_centered[row_0_indices]
    delta_t_x_measured = -np.mean(np.diff(row_0_centered_delays))

    # Y-axis: calculate delta_t per spacing unit
    col_0_indices = [0, 4, 8, 12]
    col_0_centered_delays = time_delays_centered[col_0_indices]
    delta_t_y_measured = -np.mean(np.diff(col_0_centered_delays))

    # Calculate angles
    aoa_x_calculated = apply_trig_angle(delta_t_x_measured, config.wavelength, config.speed_of_sound)
    aoa_y_calculated = apply_trig_angle(delta_t_y_measured, config.wavelength, config.speed_of_sound)

    # Calculate 3D direction vector
    dir_x = np.sin(aoa_x_calculated)
    dir_y = np.sin(aoa_y_calculated)
    sin_sum_sq = dir_x**2 + dir_y**2

    if sin_sum_sq <= 1.0:
        dir_z = np.sqrt(1.0 - sin_sum_sq)
    else:
        norm = np.sqrt(sin_sum_sq)
        dir_x /= norm
        dir_y /= norm
        dir_z = 0.0

    calculated_direction_vector = np.array([dir_x, dir_y, dir_z])

    # Theoretical values
    theoretical_angle_x = np.arctan2(source_pos[0], source_pos[2])
    theoretical_angle_y = np.arctan2(source_pos[1], source_pos[2])
    theoretical_direction_vector = source_pos / np.linalg.norm(source_pos)

    return {
        'aoa_x': aoa_x_calculated,
        'aoa_y': aoa_y_calculated,
        'calculated_direction': calculated_direction_vector,
        'theoretical_angle_x': theoretical_angle_x,
        'theoretical_angle_y': theoretical_angle_y,
        'theoretical_direction': theoretical_direction_vector,
        'distances': distances
    }


# Initialize configuration and mic positions
config = MicArrayConfig(frequency=3500, N=4)
mic_positions = generate_mic_positions(config)
mic_positions_3d = np.column_stack([mic_positions, np.zeros(len(mic_positions))])

# Initial source position
initial_source = np.array([50.0, 30.0, 100.0])

# Create figure with subplots
fig = plt.figure(figsize=(18, 8))

# Left subplot: Array closeup with unit vectors
ax_array = fig.add_subplot(121, projection='3d')

# Right subplot: 3D full view
ax_3d = fig.add_subplot(122, projection='3d')

# Adjust layout to make room for sliders
plt.subplots_adjust(bottom=0.25)

# Create sliders
ax_slider_x = plt.axes([0.15, 0.15, 0.3, 0.03])
ax_slider_y = plt.axes([0.15, 0.10, 0.3, 0.03])
ax_slider_z = plt.axes([0.15, 0.05, 0.3, 0.03])

slider_x = Slider(ax_slider_x, 'Source X (m)', -100.0, 100.0, valinit=initial_source[0], valstep=1.0)
slider_y = Slider(ax_slider_y, 'Source Y (m)', -100.0, 100.0, valinit=initial_source[1], valstep=1.0)
slider_z = Slider(ax_slider_z, 'Source Z (m)', 10.0, 200.0, valinit=initial_source[2], valstep=1.0)


def update(val):
    """Update the plots when sliders change"""
    # Get current source position
    source_pos = np.array([slider_x.val, slider_y.val, slider_z.val])

    # Calculate angles
    results = calculate_aoa_from_source(source_pos, mic_positions, config)

    # Clear both axes
    ax_array.clear()
    ax_3d.clear()

    # ===== LEFT SUBPLOT: Array closeup =====
    # Plot microphones
    ax_array.scatter(mic_positions[:, 0], mic_positions[:, 1], np.zeros(len(mic_positions)),
                     s=200, c='blue', marker='o', edgecolors='black', linewidth=2,
                     label='Microphones', depthshade=True)

    # Calculate and draw unit vectors from each mic toward source
    for i, mic_pos in enumerate(mic_positions):
        mic_pos_3d = np.array([mic_pos[0], mic_pos[1], 0])
        direction_vector = source_pos - mic_pos_3d
        direction_magnitude = np.linalg.norm(direction_vector)

        if direction_magnitude > 0:
            unit_vector = direction_vector / direction_magnitude
            arrow_scale = np.ptp(mic_positions[:, 0]) * 0.8

            ax_array.quiver(mic_pos[0], mic_pos[1], 0,
                           unit_vector[0] * arrow_scale,
                           unit_vector[1] * arrow_scale,
                           unit_vector[2] * arrow_scale,
                           color='red', alpha=0.7, arrow_length_ratio=0.2, linewidth=2)

    # Add microphone labels
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

    # Set labels and title
    ax_array.set_xlabel('X (m)', fontsize=10)
    ax_array.set_ylabel('Y (m)', fontsize=10)
    ax_array.set_zlabel('Z (m)', fontsize=10)
    ax_array.set_title(f'Array Closeup: Direction Vectors\n' +
                      f'Calculated AoA: X={np.degrees(results["aoa_x"]):.2f}°, Y={np.degrees(results["aoa_y"]):.2f}°\n' +
                      f'Theoretical: X={np.degrees(results["theoretical_angle_x"]):.2f}°, Y={np.degrees(results["theoretical_angle_y"]):.2f}°',
                      fontsize=10, fontweight='bold')

    ax_array.legend(fontsize=9, loc='upper left')

    # Adjust axis limits
    array_margin = np.ptp(mic_positions[:, 0]) * 1.3
    ax_array.set_xlim(mic_positions[:, 0].min() - array_margin*0.3, mic_positions[:, 0].max() + array_margin*0.3)
    ax_array.set_ylim(mic_positions[:, 1].min() - array_margin*0.3, mic_positions[:, 1].max() + array_margin*0.3)
    ax_array.set_zlim(0, array_margin)
    ax_array.view_init(elev=25, azim=45)

    # ===== RIGHT SUBPLOT: 3D full view =====
    # Plot microphones
    ax_3d.scatter(mic_positions[:, 0], mic_positions[:, 1], np.zeros(len(mic_positions)),
                  s=100, c='blue', marker='o', edgecolors='black', linewidth=1.5,
                  label='Microphone Array', depthshade=True)

    # Plot source
    ax_3d.scatter(source_pos[0], source_pos[1], source_pos[2],
                  s=400, c='green', marker='*', edgecolors='black', linewidth=2,
                  label=f'Source ({source_pos[0]:.0f}, {source_pos[1]:.0f}, {source_pos[2]:.0f})', depthshade=True)

    # Draw direction vectors from each mic to source
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
    ax_3d.plot_surface(xx, yy, zz, alpha=0.1, color='cyan', label='Array Plane')

    # Draw line from array center to source
    array_center = np.mean(mic_positions, axis=0)
    array_center_3d = np.array([array_center[0], array_center[1], 0])
    ax_3d.plot([array_center_3d[0], source_pos[0]],
               [array_center_3d[1], source_pos[1]],
               [array_center_3d[2], source_pos[2]],
               'g--', linewidth=2, alpha=0.7, label='Center to Source')

    # Labels and title
    ax_3d.set_xlabel('X Position (m)', fontsize=10)
    ax_3d.set_ylabel('Y Position (m)', fontsize=10)
    ax_3d.set_zlabel('Z Position (m)', fontsize=10)

    # Calculate error
    error_x = np.degrees(results["aoa_x"] - results["theoretical_angle_x"])
    error_y = np.degrees(results["aoa_y"] - results["theoretical_angle_y"])

    ax_3d.set_title(f'3D View: Array and Source\n' +
                    f'Error: X={error_x:.2f}°, Y={error_y:.2f}°',
                    fontsize=11, fontweight='bold')
    ax_3d.legend(fontsize=8, loc='upper left')

    # Set dynamic axis limits based on source position
    max_range = max(abs(source_pos[0]), abs(source_pos[1]), source_pos[2])
    ax_3d.set_xlim([-max_range*0.6, max_range*0.6])
    ax_3d.set_ylim([-max_range*0.6, max_range*0.6])
    ax_3d.set_zlim([0, max_range*1.2])
    ax_3d.view_init(elev=20, azim=45)

    fig.canvas.draw_idle()


# Connect sliders to update function
slider_x.on_changed(update)
slider_y.on_changed(update)
slider_z.on_changed(update)

# Initial plot
update(None)

plt.suptitle('Interactive AoA Visualization - Move Source with Sliders', fontsize=14, fontweight='bold', y=0.98)
plt.show()
