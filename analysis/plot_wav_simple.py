import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

# Load audio file
audio_file = 'Binary_Drone_Audio_yes_drone_B_S2_D1_067-bebop_000_.wav'
fs, audio_data = read(audio_file)

# Calculate time vector
duration = len(audio_data) / fs
t = np.arange(len(audio_data)) / fs  # Time in seconds

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Plot waveform
ax.plot(t, audio_data, linewidth=0.5, color='blue')

# Add labels and title
ax.set_xlabel('Time (s)', fontsize=12, weight='bold')
ax.set_ylabel('Amplitude', fontsize=12, weight='bold')
ax.set_title(f'Waveform: {audio_file}\nSample Rate: {fs} Hz, Duration: {duration:.3f} s',
             fontsize=14, weight='bold')

# Add grid
ax.grid(True, which='both', alpha=0.3)
ax.minorticks_on()

# Format axis
ax.set_xlim(0, duration)

# Add information text box
info_text = f'Samples: {len(audio_data)}\nMax amplitude: {np.max(np.abs(audio_data))}\nRMS: {np.sqrt(np.mean(audio_data**2)):.2f}'
ax.text(0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Tight layout
plt.tight_layout()

# Show plot
print("=" * 70)
print("WAVEFORM VISUALIZATION")
print("=" * 70)
print(f"File: {audio_file}")
print(f"Sample rate: {fs} Hz")
print(f"Duration: {duration:.3f} seconds")
print(f"Number of samples: {len(audio_data)}")
print(f"Max amplitude: {np.max(np.abs(audio_data))}")
print(f"Min amplitude: {np.min(audio_data)}")
print(f"RMS: {np.sqrt(np.mean(audio_data**2)):.2f}")
print("=" * 70)
print("\nClose the plot window to exit.")

plt.show()
