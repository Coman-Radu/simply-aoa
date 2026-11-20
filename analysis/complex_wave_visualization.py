import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 100000  # Sampling frequency (Hz)
duration = 0.001  # Duration (seconds)
t = np.linspace(0, duration, int(fs * duration))

# Generate complex sound wave (Gaussian-modulated complex exponential)
carrier_freq = 10000  # Hz
envelope = np.exp(-((t - duration/2) / (duration/10))**2)  # Gaussian envelope
complex_signal = envelope * np.exp(1j * 2 * np.pi * carrier_freq * t)

real_part = np.real(complex_signal)
imaginary_part = np.imag(complex_signal)
magnitude = np.abs(complex_signal)

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Real part
axes[0].plot(t * 1000, real_part, 'b-', linewidth=0.8)
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Real Part')
axes[0].grid(True, alpha=0.3)

# Plot 2: Imaginary part
axes[1].plot(t * 1000, imaginary_part, 'r-', linewidth=0.8)
axes[1].set_ylabel('Amplitude')
axes[1].set_title('Imaginary Part')
axes[1].grid(True, alpha=0.3)

# Plot 3: Magnitude (envelope)
axes[2].plot(t * 1000, magnitude, 'g-', linewidth=1.5, label='Magnitude')
axes[2].plot(t * 1000, real_part, 'b-', alpha=0.3, linewidth=0.5, label='Real Part')
axes[2].set_ylabel('Amplitude')
axes[2].set_title('Magnitude (Envelope)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Plot 4: Complex plane representation
# axes[3].plot(real_part, imaginary_part, 'purple', linewidth=0.8, alpha=0.6)
# axes[3].set_xlabel('Real Part')
# axes[3].set_ylabel('Imaginary Part')
# axes[3].set_title('Complex Plane Representation')
# axes[3].grid(True, alpha=0.3)
# axes[3].axis('equal')

axes[2].set_xlabel('Time (ms)')

plt.tight_layout()
plt.show()
