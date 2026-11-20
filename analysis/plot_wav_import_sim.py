import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.io.wavfile import read
from scipy.interpolate import interp1d

# Parameters
f = 600  # Hz (signal frequency)
c = 343   # m/s (speed of sound)
d = 0.049   # m (mic spacing)
theta = 30 * np.pi / 180  # angle in radians
gamma = 0 * np.pi / 180    # angle in rads
s_x_dis = 20  # x distance from origin to source
s_y_dis = s_x_dis*np.tan(theta) # y distance to source
s_z_dis = s_y_dis*np.tan(gamma) # z distance to source


def bandpass(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# Read source signal
fs, s1 = read('Binary_Drone_Audio_yes_drone_B_S2_D1_067-bebop_000_.wav')

plt.plot(s1)

no_samples = len(s1)
duration = no_samples/fs

# Define microphone positions in 3D space (linear array along x-axis)
mic_positions = np.array([
    [-d, 0, 0],  # M1: left
    [0, 0, 0],   # M2: center (reference)
    [d, 0, 0]    # M3: right
])

# Define source position in 3D space
source_position = np.array([s_x_dis, s_y_dis, s_x_dis])  # 10m away at 30deg azimuth and 30deg elevation

# Calculate distances from source to each mic
distances = np.array([np.linalg.norm(source_position - mic_pos) for mic_pos in mic_positions])

# Calculate time for wave to reach each mic
arrival_times = distances / c

# Time delays relative to first arrival (earliest mic)
time_delays = arrival_times - np.min(arrival_times)

# Create received signals by shifting the source signal in time
# Each mic records the wave when it arrives
s1_received = np.zeros_like(s1, dtype=float)
s2_received = np.zeros_like(s1, dtype=float)
s3_received = np.zeros_like(s1, dtype=float)

# Convert time delays to sample delays
sample_delays = time_delays * fs

# Shift signals using interpolation (more accurate than np.roll)
t_samples = np.arange(len(s1))

for i, (signal_out, delay) in enumerate([(s1_received, sample_delays[0]),
                                           (s2_received, sample_delays[1]),
                                           (s3_received, sample_delays[2])]):
    # Shift the signal by the delay
    interp_func = interp1d(t_samples, s1, bounds_error=False, fill_value=0)
    signal_out[:] = interp_func(t_samples - delay)

# Bandpass filter first
s1_filtered = bandpass(s1_received, f - 500, f + 500, fs)
s2_filtered = bandpass(s2_received, f - 500, f + 500, fs)
s3_filtered = bandpass(s3_received, f - 500, f + 500, fs)

# Now apply Hilbert transform to the filtered signals
analytic_s1 = hilbert(s1_filtered)
analytic_s2 = hilbert(s2_filtered)
analytic_s3 = hilbert(s3_filtered)

# Create time vector based on actual signal length
t = np.arange(len(s1)) / fs

# Cross-correlation approach for time delay estimation
from scipy.signal import correlate

corr12 = correlate(s1_filtered, s2_filtered, mode='same')
corr32 = correlate(s3_filtered, s2_filtered, mode='same')

# Find peak lag (in samples)
lag12_samples = np.argmax(corr12) - len(s2_filtered)//2
lag32_samples = np.argmax(corr32) - len(s2_filtered)//2

# Convert to time delays
lag12 = lag12_samples / fs
lag32 = lag32_samples / fs

# Calculate angles from time delays
theta12_corr = np.arcsin(np.clip(c * lag12 / d, -1, 1)) * 180/np.pi
theta32_corr = np.arcsin(np.clip(c * lag32 / d, -1, 1)) * 180/np.pi

print(f"\n=== Cross-correlation results ===")
print(f"Actual time delays: {time_delays}")
print(f"Expected tau12: {time_delays[0] - time_delays[1]:.6f} s")
print(f"Expected tau32: {time_delays[2] - time_delays[1]:.6f} s")
print(f"Estimated lag12: {lag12:.6f} s")
print(f"Estimated lag32: {lag32:.6f} s")
print(f"Angle from M1-M2: {theta12_corr:.2f}°")
print(f"Angle from M3-M2: {theta32_corr:.2f}°")
print(f"Expected angle: {theta*180/np.pi:.2f}°")



# Extract phases (unwrapped for continuity)
phi1 = np.unwrap(np.angle(analytic_s1))
phi2 = np.unwrap(np.angle(analytic_s2))
phi3 = np.unwrap(np.angle(analytic_s3))

# Plot time vs. phase
fig = plt.figure(figsize=(10, 6))
plt.plot(t, phi1, label='M1 (leading phase)', color='blue')
plt.plot(t, phi2, label='M2 (reference)', color='green')
plt.plot(t, phi3, label='M3 (lagging phase)', color='red')
plt.xlabel('Time (ms)')
plt.ylabel('Phase (radians)')
plt.title(f'Instantaneous Phase vs. Time (θ = 30°, f = {f} Hz)')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.minorticks_on()

delta_phi1_phi2 = phi1 - phi2
delta_phi3_phi2 = phi3 - phi2

tau12_est = delta_phi1_phi2/(2*np.pi*f)
tau32_est = delta_phi3_phi2/(2*np.pi*f)

theta32_est = np.arcsin(c*tau32_est/d)
theta32_est_deg = theta32_est*180/np.pi

theta12_est = np.arcsin(c*tau12_est/d)
theta12_est_deg = theta12_est*180/np.pi

plt.figure(figsize=(10, 6))

plt.plot(t, theta12_est_deg, label='θ from M1-M2 (phase)', alpha=0.5)
plt.plot(t, theta32_est_deg, label='θ from M3-M2 (phase)', alpha=0.5)

# Add cross-correlation results as horizontal lines
plt.axhline(y=theta12_corr, color='blue', linestyle='--', linewidth=2, label=f'θ M1-M2 (corr): {theta12_corr:.1f}°')
plt.axhline(y=theta32_corr, color='red', linestyle='--', linewidth=2, label=f'θ M3-M2 (corr): {theta32_corr:.1f}°')
plt.axhline(y=theta*180/np.pi, color='green', linestyle=':', linewidth=2, label=f'True θ: {theta*180/np.pi:.1f}°')

plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title(f'Estimated Angle vs. Time (f = {f} Hz)')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.minorticks_on()

# Add ESC key handler to close plot
def on_key(event):
    if event.key == 'escape':
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
