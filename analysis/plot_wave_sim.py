import numpy as np
import matplotlib.pyplot as plt

# Parameters
f = 1000  # Hz
c = 343   # m/s
d = 0.1   # m
theta = 30 * np.pi / 180  # radians
lambda_ = c / f
tau = (d * np.sin(theta)) / c  # time delay between adjacent mics

# Time vector for 2 periods
t = np.linspace(0, 2/f, 1000)

# Signals (amplitude 1 for simplicity)
s1 = np.sin(2 * np.pi * f * (t + tau))  # M1: advanced by tau (closer)
s2 = np.sin(2 * np.pi * f * t)           # M2: reference
s3 = np.sin(2 * np.pi * f * (t - tau))   # M3: delayed by tau

c1 = np.correlate(s1, s2, mode='full')   # correlate M1 - M2
c2 = np.correlate(s3, s2, mode='full')   # correlate M2 - M3

# Create lag axis for cross-correlation
lags = np.arange(-len(s2) + 1, len(s1))
lag_time = lags * (t[1] - t[0]) * 1000  # convert to ms

# Plot cross-correlations
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Subplot 1: Cross-correlation
ax1.plot(lag_time, c1, label=f'M1-M2 (peak at τ ≈ {tau*1000:.3f} ms)')
ax1.plot(lag_time, c2, label=f'M2-M3 (peak at τ ≈ {-tau*1000:.3f} ms)')
ax1.axvline(tau * 1000, color='r', linestyle='--', alpha=0.5, label='Expected delay red')
ax1.axvline(-tau * 1000, color='b', linestyle='--', alpha=0.5, label='Expected delay blue')
ax1.set_xlabel('Lag (ms)')
ax1.set_ylabel('Cross-correlation')
ax1.set_title('Cross-correlation showing time delay')
ax1.legend()
ax1.grid(True)

# Subplot 2: Original signals
#ax2.plot(t * 1000, s1, label='M1 (phase ≈ +0.915 rad)')
plt.plot(t * 1000, s2, label='M2 (phase = 0)')
#plt.plot(t * 1000, s3, label='M3 (phase ≈ -0.915 rad)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title(f'Simulated Signals for θ = 30°, Phase Diff ≈ ±{2*np.pi*f*tau:.3f} rad')
plt.legend()
plt.grid(True)

# Save to Pictures folder
plt.savefig('Pictures/phase_sim.png', dpi=150, bbox_inches='tight')
print("Plot saved to Pictures/phase_sim.png")

# Add ESC key handler to close plot
def on_key(event):
    if event.key == 'escape':
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()