import numpy as np
import matplotlib.pyplot as plt

# Parameters
f = 8000  # Frequency: 8 kHz
c = 343   # Speed of sound in air (m/s)
wavelength = c / f  # Wavelength

# Microphone array setup (linear array)
num_mics = 16
mic_spacing = wavelength / 2  # Half-wavelength spacing
mic_positions = np.array([i * mic_spacing for i in range(num_mics)])

# Source direction (angle in degrees)
source_angle = 30  # Acoustic source at 30 degrees

# Beamforming: scan angles from -90 to 90 degrees
scan_angles = np.linspace(-90, 90, 361)
beam_pattern = np.zeros(len(scan_angles))

# Calculate beam pattern using delay-and-sum beamforming
# First, calculate the steering vector for the desired source angle
source_angle_rad = np.radians(source_angle)
steering_delays = mic_positions * np.sin(source_angle_rad) / c
steering_phase_shifts = 2 * np.pi * f * steering_delays

for idx, angle in enumerate(scan_angles):
    angle_rad = np.radians(angle)

    # Calculate phase differences for each microphone at scan angle
    # Assuming far-field plane wave
    delays = mic_positions * np.sin(angle_rad) / c
    phase_shifts = 2 * np.pi * f * delays

    # Apply steering (conjugate of steering vector) and sum
    # This steers the beam to focus at source_angle
    array_response = np.sum(np.exp(1j * (phase_shifts - steering_phase_shifts)))

    # Beam pattern is the magnitude (normalized)
    beam_pattern[idx] = np.abs(array_response)

# Normalize beam pattern
beam_pattern = beam_pattern / np.max(beam_pattern)

# Convert to dB scale
beam_pattern_dB = 20 * np.log10(beam_pattern + 1e-10)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Cartesian plot
ax1.plot(scan_angles, beam_pattern_dB, 'b-', linewidth=2)
ax1.axvline(x=source_angle, color='r', linestyle='--', label=f'Source at {source_angle}°')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('Angle (degrees)', fontsize=12)
ax1.set_ylabel('Normalized Gain (dB)', fontsize=12)
ax1.set_title(f'Beamforming Pattern - {f/1000} kHz\n{num_mics}-element Linear Array', fontsize=14)
ax1.set_ylim([-40, 5])
ax1.legend()

# Plot 2: Polar plot
ax2 = plt.subplot(122, projection='polar')
angles_rad = np.radians(scan_angles)
# Shift angles for polar plot (0° at top, clockwise)
angles_polar = np.radians(90 - scan_angles)
ax2.plot(angles_polar, beam_pattern, 'b-', linewidth=2)
ax2.set_theta_zero_location('N')
ax2.set_theta_direction(-1)
ax2.set_title(f'Polar Beam Pattern\n{f/1000} kHz Signal', fontsize=14, pad=20)
ax2.set_ylim([0, 1])
ax2.grid(True)

plt.tight_layout()

# Save figure to Pictures directory
import os
save_path = os.path.join('Pictures', f'beamforming_pattern_{f//1000}kHz_{num_mics}mics_{source_angle}deg.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {save_path}")

# Print array information
print(f"\nBeamforming Configuration:")
print(f"  Frequency: {f/1000} kHz")
print(f"  Wavelength: {wavelength*100:.2f} cm")
print(f"  Number of microphones: {num_mics}")
print(f"  Microphone spacing: {mic_spacing*100:.2f} cm")
print(f"  Array length: {(num_mics-1)*mic_spacing*100:.2f} cm")
print(f"  Beam width (-3dB): ~{2*np.degrees(np.arcsin(0.886*wavelength/(num_mics*mic_spacing))):.1f}°")

plt.show()
