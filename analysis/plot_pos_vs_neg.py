#!/usr/bin/env python3
"""
plot_analytic_vs_real.py

Create a real-valued test signal, compute its analytic signal (Hilbert transform),
and plot time-domain and frequency-domain representations showing positive and
negative frequencies.

Save and run locally; this script is NOT executed here.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# -------------------------
# Parameters
# -------------------------
fs = 2000.0         # sampling frequency (Hz)
T = 0.5             # duration (s)
t = np.arange(0, T, 1/fs)
N = len(t)

# Create a real test signal composed of two cosines (50 Hz and 150 Hz)
f1 = 50.0
f2 = 150.0
x = 1.0 * np.cos(2*np.pi*f1*t) + 0.6 * np.cos(2*np.pi*f2*t)

# Compute analytic signal using Hilbert transform
x_a = hilbert(x)   # complex analytic signal: real(x_a)=x, imag(x_a)=Hilbert{x}

plt.figure()

plt.plot(np.unwrap(np.angle(x_a)))

plt.show()
# -------------------------
# Frequency-domain setup
# -------------------------
# Use a larger FFT length for smoother spectrum visualization
Nfft = 8192
X = np.fft.fft(x, Nfft)
Xa = np.fft.fft(x_a, Nfft)

# Frequency vector in Hz
freqs = np.fft.fftfreq(Nfft, d=1.0/fs)    # ranges from 0..fs/2 then -fs/2..0 (unsorted)
# Shifted versions (so zero-frequency is centered)
freqs_shift = np.fft.fftshift(freqs)
X_shift = np.fft.fftshift(X)
Xa_shift = np.fft.fftshift(Xa)

# Magnitudes (linear) and in dB (optional)
mag_X = np.abs(X_shift)
mag_Xa = np.abs(Xa_shift)
# Small epsilon to avoid log10(0)
eps = 1e-20
mag_X_db = 20 * np.log10(mag_X + eps)
mag_Xa_db = 20 * np.log10(mag_Xa + eps)

# -------------------------
# Plotting
# -------------------------
plt.rcParams.update({'figure.max_open_warning': 0})

fig = plt.figure(figsize=(12, 10))

# Time-domain: real signal and analytic signal (real & imag)
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(t, x, label='real signal x(t)', linewidth=1)
ax1.plot(t, np.real(x_a), '--', label='real(x_a) (should equal x)', alpha=0.7)
ax1.plot(t, np.imag(x_a), label='imag(x_a) = Hilbert{x(t)}', linewidth=1, alpha=0.9)

ax1.plot(t, np.angle(x_a), label="Angle Hilbert", linewidth=1, alpha=0.9)

ax1.set_xlim(0, 0.05)  # zoom first 50 ms for clarity
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Amplitude')
ax1.set_title('Time domain: real signal and analytic signal (real & imaginary parts)')
ax1.legend(loc='upper right')
ax1.grid(True)

# Frequency-domain (centered) - magnitude (linear)
ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(freqs_shift, mag_X, label='|FFT{x(t)}| (centered)', linewidth=1)
ax2.plot(freqs_shift, mag_Xa, label='|FFT{analytic x_a(t)}| (centered)', linewidth=1, alpha=0.8)
ax2.set_xlim(-400, 400)
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Magnitude')
ax2.set_title('Centered magnitude spectra (linear scale)')
ax2.legend(loc='upper right')
ax2.grid(True)

# Frequency-domain (centered) - magnitude (dB) with focus on negative vs positive frequencies
ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(freqs_shift, mag_X_db, label='FFT{x} (dB)', linewidth=1)
ax3.plot(freqs_shift, mag_Xa_db, label='FFT{analytic x_a} (dB)', linewidth=1, alpha=0.8)
ax3.set_xlim(-400, 400)
ax3.set_ylim(np.max(mag_X_db)-80, np.max(mag_X_db)+5)  # zoom vertically
ax3.set_xlabel('Frequency [Hz]')
ax3.set_ylabel('Magnitude [dB]')
ax3.set_title('Centered magnitude spectra (dB) — observe negative vs positive frequencies')
ax3.legend(loc='upper right')
ax3.grid(True)

plt.tight_layout()

 # Save to Pictures folder
plt.savefig('Pictures/pos_vs_ned.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to Pictures/pos_vs_neg.png")

    # Add ESC key handler to close plot
def on_key(event):
    if event.key == 'escape':
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

# -------------------------
# Notes (printed when script runs)
# -------------------------
# - The real-valued x(t) will show symmetric spectral lines at +f1, -f1 and +f2, -f2.
# - The analytic signal x_a(t) is complex and ideally should have (almost) zero energy
#   in the negative-frequency side of the spectrum. In practice numerical effects
#   and the discrete Hilbert transform can leave tiny residuals.
# - In continuous-time theory the analytic signal doubles the positive-frequency
#   spectral components (except at DC and Nyquist) because the negative-frequency
#   components were removed; you may see that in the plotted magnitudes.
#
# Tips:
# - If you want a one-sided plot (positive frequencies only), take freqs[freqs>=0]
#   and corresponding FFT bins. For analytic signals, the one-sided spectrum will
#   contain the main energy and show the doubling described above.
# - To inspect instantaneous amplitude (envelope) and phase:
#       envelope = np.abs(x_a)
#       inst_phase = np.unwrap(np.angle(x_a))
#       inst_freq = np.diff(inst_phase) * fs / (2*np.pi)
#
# Save this file and run:
#   python plot_analytic_vs_real.py
#
# End of script
