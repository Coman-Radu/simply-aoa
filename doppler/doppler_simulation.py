# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, hilbert
from matplotlib.animation import FuncAnimation
import sys
import io

# Set stdout encoding to UTF-8 for proper character display
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Physical constants
SPEED_OF_SOUND = 343.0  # m/s
SAMPLE_RATE = 48000  # Hz
MIC_SPACING = 0.04  # meters

# Source parameters
SOURCE_VELOCITY = 10.0  # m/s (horizontal)
SOURCE_Y = 50.0  # meters (constant y position)
SOURCE_DISTANCE = 100.0  # meters (initial distance from array center)
SOURCE_FREQUENCY = 3000.0  # Hz (drone-like tone)

# Simulation parameters
SIMULATION_DURATION = 20.0  # seconds
BUFFER_DURATION = 0.5  # seconds per DOA estimate
NUM_MICS = 4

# Microphone positions (on x-axis)
mic_positions = np.array([
    [-1.5 * MIC_SPACING, 0],  # Mic 0
    [-0.5 * MIC_SPACING, 0],  # Mic 1
    [0.5 * MIC_SPACING, 0],   # Mic 2
    [1.5 * MIC_SPACING, 0]    # Mic 3
])

print("=" * 80)
print("DOPPLER EFFECT SIMULATION - MOVING SOURCE WITH TDOA/DOA ESTIMATION")
print("=" * 80)
print(f"Array: {NUM_MICS} microphones, spacing = {MIC_SPACING*1000:.1f} mm")
print(f"Source: {SOURCE_FREQUENCY:.0f} Hz tone, velocity = {SOURCE_VELOCITY:.1f} m/s")
print(f"Source path: y = {SOURCE_Y:.1f} m (horizontal motion)")
print(f"Initial distance: {SOURCE_DISTANCE:.1f} m")
print(f"Simulation duration: {SIMULATION_DURATION:.1f} s")
print(f"Buffer duration: {BUFFER_DURATION:.1f} s")
print("=" * 80)
print()


def source_position(t):
    """
    Calculate source position at time t.
    Source starts at x such that initial distance is SOURCE_DISTANCE.

    Args:
        t: Time in seconds

    Returns:
        np.array: [x, y] position
    """
    # Initial x position: source starts to the left
    x_initial = -np.sqrt(SOURCE_DISTANCE**2 - SOURCE_Y**2)
    x = x_initial + SOURCE_VELOCITY * t
    y = SOURCE_Y
    return np.array([x, y])


def calculate_true_doa(source_pos):
    """
    Calculate true direction of arrival (azimuth).
    0° = North (+y), 90° = East (+x), clockwise

    Args:
        source_pos: [x, y] position of source

    Returns:
        azimuth: DOA in degrees
    """
    # Vector from array center to source
    direction = source_pos / np.linalg.norm(source_pos)

    # Azimuth: 0° = North (+y), 90° = East (+x)
    azimuth = np.degrees(np.arctan2(direction[0], direction[1]))
    if azimuth < 0:
        azimuth += 360

    return azimuth


def calculate_doppler_shift(source_pos, source_vel, mic_pos):
    """
    Calculate Doppler-shifted frequency for a microphone.

    Args:
        source_pos: [x, y] source position
        source_vel: [vx, vy] source velocity
        mic_pos: [x, y] microphone position

    Returns:
        f_observed: Doppler-shifted frequency
    """
    # Vector from source to microphone
    r_vec = mic_pos - source_pos
    r = np.linalg.norm(r_vec)
    r_hat = r_vec / r  # Unit vector

    # Radial velocity (positive = approaching)
    v_radial = np.dot(source_vel, r_hat)

    # Doppler formula: f_obs = f_source * c / (c - v_radial)
    f_observed = SOURCE_FREQUENCY * SPEED_OF_SOUND / (SPEED_OF_SOUND - v_radial)

    return f_observed



def generate_signal_at_mic(t_start, duration, mic_pos, add_noise=True):
    """
    Generate signal received at a microphone from a moving source.
    Includes Doppler shift and propagation delay.

    Args:
        t_start: Start time of signal
        duration: Duration of signal
        mic_pos: [x, y] microphone position
        add_noise: Whether to add noise

    Returns:
        signal: Audio signal
        times: Time array
    """
    num_samples = int(duration * SAMPLE_RATE)
    times = np.linspace(t_start, t_start + duration, num_samples)
    signal = np.zeros(num_samples)

    source_vel = np.array([SOURCE_VELOCITY, 0])  # Horizontal motion

    for i, t in enumerate(times):
        # Source position at emission time
        # We need to solve for emission time t_emit such that:
        # ||source_pos(t_emit) - mic_pos|| / c = t - t_emit

        # Simplified: assume emission time ≈ t - distance/c (iterative refinement)
        source_pos_t = source_position(t)
        distance = np.linalg.norm(source_pos_t - mic_pos)
        t_emit = t - distance / SPEED_OF_SOUND

        # Refine (one iteration)
        source_pos_emit = source_position(t_emit)
        distance = np.linalg.norm(source_pos_emit - mic_pos)
        t_emit = t - distance / SPEED_OF_SOUND

        # Calculate Doppler shift at emission time
        source_pos_emit = source_position(t_emit)
        f_doppler = calculate_doppler_shift(source_pos_emit, source_vel, mic_pos)

        # Phase at this time (accumulated phase with Doppler)
        # For simplicity, use instantaneous frequency
        phase = 2 * np.pi * f_doppler * t_emit

        # Amplitude (1/r spreading)
        distance = np.linalg.norm(source_pos_emit - mic_pos)
        amplitude = 100.0 / distance  # Scale factor

        signal[i] = amplitude * np.sin(phase)

    # Add noise
    if add_noise:
        noise_level = 0.1 * np.std(signal)
        signal += noise_level * np.random.randn(num_samples)

    return signal, times


def estimate_tdoa_cross_correlation(signal1, signal2, ref_idx=0):
    """
    Estimate TDOA between two signals using cross-correlation.

    Args:
        signal1: Reference signal
        signal2: Signal to compare
        ref_idx: Reference index (unused, for compatibility)

    Returns:
        tdoa: Time delay in seconds (positive if signal2 arrives after signal1)
    """
    # Cross-correlation
    correlation = correlate(signal2, signal1, mode='same')
    lags = np.arange(-len(signal1)//2, len(signal1)//2)

    # Find peak
    peak_idx = np.argmax(np.abs(correlation))
    peak_lag = lags[peak_idx]

    # Convert to time
    tdoa = peak_lag / SAMPLE_RATE

    return tdoa


def estimate_doa_from_tdoa(tdoas, method='linear'):
    """
    Estimate DOA from TDOAs using linear array geometry.

    Args:
        tdoas: Array of TDOAs relative to mic 0
        method: 'linear' or 'hyperbolic'

    Returns:
        azimuth: Estimated DOA in degrees
    """
    if method == 'linear':
        # Simple linear approximation: assume far-field plane wave
        # TDOA = (mic_x / c) * sin(theta)
        # where theta is angle from broadside (y-axis)

        # Use mic pair 0-3 (largest baseline)
        tdoa_03 = tdoas[3]
        baseline = mic_positions[3, 0] - mic_positions[0, 0]

        # sin(theta) = TDOA * c / baseline
        sin_theta = (tdoa_03 * SPEED_OF_SOUND) / baseline
        sin_theta = np.clip(sin_theta, -1, 1)

        # theta from broadside
        theta_from_broadside = np.degrees(np.arcsin(sin_theta))

        # Convert to azimuth (0° = North/+y, 90° = East/+x)
        azimuth = 90 - theta_from_broadside
        if azimuth < 0:
            azimuth += 360

        return azimuth
    else:
        # Hyperbolic method (more accurate for near-field)
        # Not implemented here for simplicity
        return 0.0


def run_simulation():
    """
    Run the complete simulation and collect results.
    """
    # Time windows for DOA estimation
    num_windows = int(SIMULATION_DURATION / BUFFER_DURATION)

    results = {
        'times': [],
        'source_positions': [],
        'true_doas': [],
        'estimated_doas': [],
        'doa_errors': [],
        'tdoas': [],
        'doppler_shifts': []
    }

    print("Running simulation...")
    print()

    for window_idx in range(num_windows):
        t_start = window_idx * BUFFER_DURATION
        t_center = t_start + BUFFER_DURATION / 2

        # Generate signals at all microphones
        signals = []
        for mic_idx in range(NUM_MICS):
            signal, times = generate_signal_at_mic(
                t_start, BUFFER_DURATION, mic_positions[mic_idx], add_noise=True
            )
            signals.append(signal)

        signals = np.array(signals)  # [num_mics, num_samples]

        # Estimate TDOAs relative to mic 0
        tdoas = np.zeros(NUM_MICS)
        for mic_idx in range(1, NUM_MICS):
            tdoas[mic_idx] = estimate_tdoa_cross_correlation(signals[0], signals[mic_idx])

        # Estimate DOA from TDOAs
        estimated_doa = estimate_doa_from_tdoa(tdoas, method='linear')

        # True values at center of window
        source_pos = source_position(t_center)
        true_doa = calculate_true_doa(source_pos)

        # Calculate Doppler shifts for each mic
        source_vel = np.array([SOURCE_VELOCITY, 0])
        doppler_shifts = []
        for mic_idx in range(NUM_MICS):
            f_doppler = calculate_doppler_shift(source_pos, source_vel, mic_positions[mic_idx])
            doppler_shifts.append(f_doppler - SOURCE_FREQUENCY)

        # DOA error
        doa_error = estimated_doa - true_doa
        # Handle wraparound
        if doa_error > 180:
            doa_error -= 360
        elif doa_error < -180:
            doa_error += 360

        # Store results
        results['times'].append(t_center)
        results['source_positions'].append(source_pos)
        results['true_doas'].append(true_doa)
        results['estimated_doas'].append(estimated_doa)
        results['doa_errors'].append(doa_error)
        results['tdoas'].append(tdoas)
        results['doppler_shifts'].append(doppler_shifts)

        # Print progress
        distance = np.linalg.norm(source_pos)
        print(f"t={t_center:5.1f}s: Src=({source_pos[0]:6.1f}, {source_pos[1]:5.1f})m, "
              f"True={true_doa:6.1f}°, Est={estimated_doa:6.1f}°, "
              f"Error={doa_error:+5.1f}°, Doppler={doppler_shifts[0]:+5.1f}Hz")

    print()
    print("Simulation complete.")
    print()

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


def plot_results(results):
    """
    Plot simulation results.
    """
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Geometry and source trajectory
    ax1 = fig.add_subplot(2, 3, 1)

    # Plot microphone array
    ax1.plot(mic_positions[:, 0], mic_positions[:, 1], 'ko', markersize=10, label='Microphones')
    for i, pos in enumerate(mic_positions):
        ax1.text(pos[0], pos[1] - 2, f'M{i}', ha='center', va='top', fontsize=8)

    # Plot source trajectory
    source_positions = results['source_positions']
    ax1.plot(source_positions[:, 0], source_positions[:, 1], 'b-', linewidth=2, alpha=0.5, label='Source trajectory')

    # Mark start and end
    ax1.plot(source_positions[0, 0], source_positions[0, 1], 'go', markersize=12, label='Start')
    ax1.plot(source_positions[-1, 0], source_positions[-1, 1], 'ro', markersize=12, label='End')

    # Mark closest approach
    distances = np.linalg.norm(source_positions, axis=1)
    closest_idx = np.argmin(distances)
    ax1.plot(source_positions[closest_idx, 0], source_positions[closest_idx, 1],
             'mo', markersize=12, label='Closest approach')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Geometry: Microphone Array and Source Trajectory')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')

    # Plot 2: True vs Estimated DOA over time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(results['times'], results['true_doas'], 'g-', linewidth=2, label='True DOA')
    ax2.plot(results['times'], results['estimated_doas'], 'b--', linewidth=2, label='Estimated DOA')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Azimuth (degrees)')
    ax2.set_title('DOA Estimation: True vs Estimated')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 180])

    # Plot 3: DOA Error over time
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(results['times'], results['doa_errors'], 'r-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('DOA Error (degrees)')
    ax3.set_title('DOA Estimation Error vs Time')
    ax3.grid(True, alpha=0.3)

    # Calculate RMS error
    rms_error = np.sqrt(np.mean(results['doa_errors']**2))
    ax3.text(0.02, 0.98, f'RMS Error: {rms_error:.2f}°',
             transform=ax3.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Doppler shift over time
    ax4 = fig.add_subplot(2, 3, 4)
    doppler_shifts = results['doppler_shifts']
    for mic_idx in range(NUM_MICS):
        ax4.plot(results['times'], doppler_shifts[:, mic_idx],
                linewidth=2, label=f'Mic {mic_idx}')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Doppler Shift (Hz)')
    ax4.set_title('Doppler Shift at Each Microphone')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: TDOA over time
    ax5 = fig.add_subplot(2, 3, 5)
    tdoas = results['tdoas']
    for mic_idx in range(1, NUM_MICS):
        ax5.plot(results['times'], tdoas[:, mic_idx] * 1e6,
                linewidth=2, label=f'M{mic_idx} - M0')
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('TDOA (us)')
    ax5.set_title('Time Difference of Arrival (relative to Mic 0)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Plot 6: Source distance and angular velocity
    ax6 = fig.add_subplot(2, 3, 6)

    # Distance
    distances = np.linalg.norm(source_positions, axis=1)
    ax6_twin = ax6.twinx()

    line1 = ax6.plot(results['times'], distances, 'b-', linewidth=2, label='Distance')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Distance (m)', color='b')
    ax6.tick_params(axis='y', labelcolor='b')
    ax6.grid(True, alpha=0.3)

    # Angular velocity (approximate)
    true_doas = results['true_doas']
    dt = results['times'][1] - results['times'][0]
    angular_velocity = np.gradient(true_doas, dt)

    line2 = ax6_twin.plot(results['times'], angular_velocity, 'r-', linewidth=2, label='Angular velocity')
    ax6_twin.set_ylabel('Angular Velocity (°/s)', color='r')
    ax6_twin.tick_params(axis='y', labelcolor='r')

    ax6.set_title('Source Distance and Angular Velocity')

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig('doppler_simulation_results.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'doppler_simulation_results.png'")
    plt.show()


def analyze_results(results):
    """
    Print statistical analysis of results.
    """
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # DOA errors
    doa_errors = results['doa_errors']
    rms_error = np.sqrt(np.mean(doa_errors**2))
    mean_error = np.mean(doa_errors)
    max_error = np.max(np.abs(doa_errors))

    print(f"DOA Estimation Errors:")
    print(f"  Mean error:  {mean_error:+6.2f}°")
    print(f"  RMS error:   {rms_error:6.2f}°")
    print(f"  Max error:   {max_error:6.2f}°")
    print()

    # Doppler shifts
    doppler_shifts = results['doppler_shifts']
    max_doppler = np.max(np.abs(doppler_shifts))
    mean_doppler = np.mean(np.abs(doppler_shifts))

    print(f"Doppler Shifts:")
    print(f"  Mean (abs):  {mean_doppler:6.2f} Hz")
    print(f"  Max (abs):   {max_doppler:6.2f} Hz")
    print(f"  Percentage:  {100 * max_doppler / SOURCE_FREQUENCY:.2f}%")
    print()

    # Angular velocity
    true_doas = results['true_doas']
    dt = results['times'][1] - results['times'][0]
    angular_velocity = np.gradient(true_doas, dt)
    max_angular_vel = np.max(np.abs(angular_velocity))

    print(f"Source Motion:")
    print(f"  Max angular velocity: {max_angular_vel:.2f} °/s")
    print(f"  Angular displacement during buffer ({BUFFER_DURATION}s): {max_angular_vel * BUFFER_DURATION:.2f}°")
    print()

    # Closest approach
    source_positions = results['source_positions']
    distances = np.linalg.norm(source_positions, axis=1)
    closest_idx = np.argmin(distances)
    closest_dist = distances[closest_idx]
    closest_time = results['times'][closest_idx]

    print(f"Closest Approach:")
    print(f"  Time:     {closest_time:.2f} s")
    print(f"  Distance: {closest_dist:.2f} m")
    print(f"  True DOA: {results['true_doas'][closest_idx]:.1f}°")
    print(f"  Est. DOA: {results['estimated_doas'][closest_idx]:.1f}°")
    print(f"  Error:    {results['doa_errors'][closest_idx]:+.1f}°")
    print()

    # TDOA range
    tdoas = results['tdoas']
    max_tdoa = np.max(np.abs(tdoas[:, 1:]))  # Exclude mic 0 (reference)

    print(f"TDOA Statistics:")
    print(f"  Max TDOA: {max_tdoa * 1e6:.2f} us")
    print(f"  Max TDOA (samples @ {SAMPLE_RATE} Hz): {max_tdoa * SAMPLE_RATE:.2f}")
    print()

    print("=" * 80)


if __name__ == "__main__":
    
    
    # Run simulation
    results = run_simulation()

    # Analyze results
    analyze_results(results)


    plt.plot(
        
    )
    # Plot results
    plot_results(results)
