import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.io.wavfile import read

# Parameters
f = 3500  # Hz (signal frequency)
c = 343   # m/s (speed of sound)
d = 0.049   # m (mic spacing)
theta = 30 * np.pi / 180  # angle in radians

# Time delay between adjacent mics
tau = (d * np.sin(theta)) / c  # ≈ 0.1457 ms

fs, s1 = read('Binary_Drone_Audio_yes_drone_B_S2_D1_067-bebop_000_.wav')

plt.plot(s1)

no_samples = len(s1)
duration = no_samples/fs

# Create time vector based on actual signal length
t = np.arange(len(s1)) / fs

def bandpass(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Reference analytic signal
analytic_s_ref = hilbert(s1)

# Create 4 separate vectors for the 4-mic array
# m_0 = 3*tau/2, m_1 = tau/2, m_2 = -tau/2, m_3 = -3*tau/2
analytic_s0 = analytic_s_ref * np.exp(1j*2*np.pi*f * (3*tau/2))
analytic_s1 = analytic_s_ref * np.exp(1j*2*np.pi*f * (tau/2))
analytic_s2 = analytic_s_ref * np.exp(1j*2*np.pi*f * (-tau/2))
analytic_s3 = analytic_s_ref * np.exp(1j*2*np.pi*f * (-3*tau/2))

# Extract phases (unwrapped for continuity)
phi0 = np.unwrap(np.angle(analytic_s0))
phi1 = np.unwrap(np.angle(analytic_s1))
phi2 = np.unwrap(np.angle(analytic_s2))
phi3 = np.unwrap(np.angle(analytic_s3))

# Plot time vs. phase
fig = plt.figure(figsize=(10, 6))
plt.plot(t, phi0, label='M0 (3τ/2)', color='blue')
plt.plot(t, phi1, label='M1 (τ/2)', color='green')
plt.plot(t, phi2, label='M2 (-τ/2)', color='orange')
plt.plot(t, phi3, label='M3 (-3τ/2)', color='red')

plt.xlabel('Time (s)')
plt.ylabel('Phase (radians)')
plt.title(f'Instantaneous Phase vs. Time - 4-Mic Array (θ = 30°, f = {f} Hz)')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.minorticks_on()

# Calculate phase differences and angle estimates
delta_phi01 = phi0 - phi1
delta_phi12 = phi1 - phi2
delta_phi23 = phi2 - phi3

tau01_est = delta_phi01 / (2*np.pi*f)
tau12_est = delta_phi12 / (2*np.pi*f)
tau23_est = delta_phi23 / (2*np.pi*f)

theta01_est_deg = np.arcsin(np.clip(c*tau01_est/d, -1, 1)) * 180/np.pi
theta12_est_deg = np.arcsin(np.clip(c*tau12_est/d, -1, 1)) * 180/np.pi
theta23_est_deg = np.arcsin(np.clip(c*tau23_est/d, -1, 1)) * 180/np.pi

# Plot angle estimates
plt.figure(figsize=(10, 6))
plt.plot(t, theta01_est_deg, label='M0-M1', alpha=0.7)
plt.plot(t, theta12_est_deg, label='M1-M2', alpha=0.7)
plt.plot(t, theta23_est_deg, label='M2-M3', alpha=0.7)

plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title(f'Estimated Angle vs. Time - 4-Mic Array (f = {f} Hz)')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.minorticks_on()

# Add ESC key handler to close plot
def on_key(event):
    if event.key == 'escape':
        plt.close('all')

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
