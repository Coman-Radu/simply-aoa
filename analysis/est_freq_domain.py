import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.fft import fft, fftfreq, ifft, fftshift
from dataclasses import dataclass

fs, s1 = read('Binary_Drone_Audio_yes_drone_B_S2_D1_067-bebop_000_.wav')

s1 = s1/max(abs(s1))

#s1 = s1.astype(np.float64)

N = len(s1)


t = np.linspace(0, N/fs, N)
sample = 300
s2 = np.roll(s1, sample)

N_fft = 2*len(s1) - 1

freq_s1 = fft(s1, N_fft)
freq_s2 = fft(s2, N_fft)


conv = np.conjugate(freq_s2) * freq_s1


conv_norm = conv/(np.abs(conv)+1e-10)

inv_fft = ifft(conv_norm)

inv_fft = fftshift(inv_fft)

lags = np.arange(-N_fft//2, N_fft//2)

plt.plot(lags[:-1], inv_fft.real[:-1])
plt.show()

max_index = np.argmax(inv_fft)
max_fft = inv_fft[max_index]

delay_est = lags[max_index]

print(f'max index :{max_index},\n delay_est :{delay_est}. \n')

