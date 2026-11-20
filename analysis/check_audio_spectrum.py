import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.fft import fft, fftfreq

f1= 500
f2= 1200
f3= 3000
# Read the audio file
fs, signal = read('Binary_Drone_Audio_yes_drone_B_S2_D1_067-bebop_000_.wav')





# Compute FFT
N = len(signal)

#t = np.linspace(0, N/fs, N)

#signal = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + np.sin(2*np.pi*f3*t)

yf = fft(signal)
xf = fftfreq(N, 1/fs)[:N//2]




# Find peak frequency
magnitude = 2.0/N * np.abs(yf[:N//2])
peak_idx = np.argmax(magnitude)
peak_frequency = xf[peak_idx]

# Plot spectrum
plt.figure(figsize=(12, 6))
#plt.plot(t, signal)
plt.plot(xf, magnitude)
plt.axvline(peak_frequency, color='g', linestyle='--', label=f'Peak: {peak_frequency:.2f} Hz')
plt.axvline(3500, color='r', linestyle='--', label='Target: 3500 Hz')
#plt.xlim([peak_frequency - 100, peak_frequency + 100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Drone Audio')
plt.grid(True)
plt.legend()
plt.show()

print(f"Peak frequency: {peak_frequency:.2f} Hz")
