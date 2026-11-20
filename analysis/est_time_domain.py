import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.fft import fft, fftfreq
from dataclasses import dataclass

fs, s1 = read('Binary_Drone_Audio_yes_drone_B_S2_D1_067-bebop_000_.wav')

s1 = s1/max(abs(s1))

#s1 = s1.astype(np.float64)

N = len(s1)


t = np.linspace(0, N/fs, N)
sample = 200

s2 = np.roll(s1, sample)



#plt.plot(t, s1, label = 'original')
#plt.plot(t, s2, label = 'delayed')
#plt.xlabel('time (s)')
#plt.ylabel('Amplitude')
#plt.legend()
#plt.show()

energy = np.zeros(2*sample)

for i in range(0, 2*sample):

    
    s1_delay = np.roll(s1, i)
    
    sum_s = s1_delay + s2
    
    energy[i] = sum(sum_s**2)
    
plt.plot(energy)
plt.show()