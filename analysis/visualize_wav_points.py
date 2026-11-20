import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

# Load WAV file
fs, wav_data = read('drone_monotone_3kHz.wav')

# Convert to float if necessary
if wav_data.dtype == np.int16:
    wav_data = wav_data.astype(np.float32) / 32768.0
elif wav_data.dtype == np.int32:
    wav_data = wav_data.astype(np.float32) / 2147483648.0

# If stereo, take first channel
if wav_data.ndim > 1:
    wav_data = wav_data[:, 0]

# Use only a small portion for visualization (first 5ms)
duration = 0.005  # 5ms
num_samples = int(duration * fs)
wav_data = wav_data[:num_samples]

# Time vector
t = np.arange(len(wav_data)) / fs

# Plot amplitude vectors as vertical lines from 0
fig = plt.figure(figsize=(14, 6))

# Draw vectors from (t, 0) to (t, amplitude)
for i in range(len(t)):
    plt.plot([t[i] * 1000, t[i] * 1000], [0, wav_data[i]], 'b-', alpha=0.5, linewidth=0.5)

# Mark vector endpoints
plt.scatter(t * 1000, wav_data, s=10, c='red', zorder=5, label='Vector endpoints')
plt.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.5)

plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.title('WAV File Visualization - Amplitude Vectors', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Save to Pictures folder
plt.savefig('Pictures/wav_points.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to Pictures/wav_points.png")
print(f"Sample rate: {fs} Hz")
print(f"Duration: {len(wav_data)/fs*1000:.2f} ms")
print(f"Samples: {len(wav_data)}")

# Add ESC key handler to close plot
def on_key(event):
    if event.key == 'escape':
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
