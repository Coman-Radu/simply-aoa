import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.io.wavfile import read

# Parameters
f = 3500  # Hz (signal frequency)
c = 343   # m/s (speed of sound)
d = 0.049   # m (mic spacing)
theta = 20 * np.pi / 180  # angle in radians
#fs = 44100  # Hz (sampling rate)
#duration = 0.002  # seconds (2 ms, ~2 periods)

# Time delay between adjacent mics
tau = (d * np.sin(theta)) / c  # ≈ 0.1457 ms

fs, s1 = read('Binary_Drone_Audio_yes_drone_B_S2_D1_067-bebop_000_.wav')

plt.plot(s1)

no_samples = len(s1)

duration = no_samples/fs

# Simulated signals (sinusoids with phase shifts)
t = np.linspace(0, duration, int(fs * duration))
#s1 = np.sin(2 * np.pi * f * (t + tau))  # M1: advanced
#s2 = np.sin(2 * np.pi * f * t)           # M2: reference
#s3 = np.sin(2 * np.pi * f * (t - tau))   # M3: delayed


# Compute analytic signals and instantaneous phases
analytic_s1 = hilbert(s1)*np.exp(-1j*3*np.pi*f*tau)
analytic_s2 = hilbert(s1)*np.exp(-1j*np.pi*f*tau)
analytic_s3 = hilbert(s1)*np.exp(1j*np.pi*f*tau)
analytic_s4 = hilbert(s1)*np.exp(1j*3*np.pi*f*tau)

#fs, s1 = read('Binary_Drone_Audio_yes_drone_B_S2_D1_067-bebop_000_')
#_, s2 = read('Binary_Drone_Audio_yes_drone_B_S2_D1_067-bebop_000_')
#_, s3 = read('Binary_Drone_Audio_yes_drone_B_S2_D1_067-bebop_000_')

# Create time vector based on actual signal length
t = np.arange(len(s1)) / fs

def bandpass(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)
#s1 = bandpass(s1, 3000, 4000, fs)  # Example: 900-1100 Hz
#s2 = bandpass(s2, 3000, 4000, fs)
#s3 = bandpass(s3, 3000, 4000, fs)

#analytic_s1 = bandpass(analytic_s1, 900, 1100, fs)  # Example: 900-1100 Hz
#analytic_s1 = bandpass(analytic_s2, 900, 1100, fs)
#analytic_s3 = bandpass(analytic_s3, 900, 1100, fs)

# Compute analytic signals and instantaneous phases
#analytic_s1 = hilbert(s1)
#analytic_s2 = hilbert(s2)
#analytic_s3 = hilbert(s3)

# Extract phases (unwrapped for continuity)
phi1 = np.unwrap(np.angle(analytic_s1))
phi2 = np.unwrap(np.angle(analytic_s2))
phi3 = np.unwrap(np.angle(analytic_s3))
phi4 = np.unwrap(np.angle(analytic_s4))

#phi1 = np.angle(analytic_s1)
#phi2 = np.angle(analytic_s2)
#phi3 = np.angle(analytic_s3)
phi4 = np.angle(analytic_s4)

# Plot time vs. phase
fig = plt.figure(figsize=(10, 6))

plt.plot(t, phi1, label='M1 (leading phase)', color='blue')
plt.plot(t, phi2, label='M2 (reference)', color='green')
plt.plot(t, phi3, label='M3 (lagging phase)', color='red')
plt.plot(t, phi4, label="M4", color="purple")
plt.xlabel('Time (ms)')
plt.ylabel('Phase (radians)')
plt.title(f'Instantaneous Phase vs. Time (θ = 30°, f = {f} Hz)')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.minorticks_on()
# Set more granular y-axis ticks
ax = plt.gca()
#ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
#ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi/4))

#delta_phi1_phi2 = phi1 - phi2
#delta_phi3_phi4 = phi2 - phi3

#tau12_est = delta_phi1_phi2/(2*np.pi*f)
#tau34_est = delta_phi3_phi4/(2*np.pi*f)

#theta34_est = np.arcsin(c*tau34_est/d)
#theta34_est_deg = theta34_est*180/np.pi

#theta12_est = np.arcsin(c*tau12_est/d)
#theta12_est_deg = theta12_est*180/np.pi

#Avg_theta = (theta12_est_deg + theta34_est_deg)/2

d_phi1_ph2 = phi1 - phi2
d_phi12_phi3 = d_phi1_ph2 - phi3
d_phi123_phi4 = d_phi12_phi3 - phi4

tau1234_est = d_phi123_phi4//(2*np.pi*f)
theta1234_est = np.arcsin(c*tau1234_est/d)
theta1234_est_deg = theta1234_est*180/np.pi




plt.figure(figsize=(10, 6))
#plt.plot(t, delta_phi1_phi2, label='M1 - M2 (leading phase)', color='blue')
#plt.plot(t, phi2, label='M2 (reference)', color='green')
#plt.plot(t, delta_phi3_phi2, label='M3 -M2 (lagging phase)', color='red')

#plt.plot(t, theta12_est_deg)
#plt.plot(t, theta34_est_deg)
#plt.plot(t, Avg_theta)

plt.plot(t, theta1234_est_deg)

plt.xlabel('Time (ms)')
plt.ylabel('Phase (radians)')
plt.title(f'Instantaneous Phase vs. Time (θ = 30°, f = {f} Hz)')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.minorticks_on()

# Save to Pictures folder
#plt.savefig('Pictures/phase_vs_time.png', dpi=150, bbox_inches='tight')
#print("Plot saved to Pictures/phase_vs_time.png")

# Add ESC key handler to close plot
def on_key(event):
    if event.key == 'escape':
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()