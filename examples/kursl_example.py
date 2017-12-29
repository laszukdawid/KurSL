import numpy as np
import pylab as plt
import sys

sys.path.append('..')

from kursl import KurSL

t0, t1, dt = 0, 5, 0.01
t = np.arange(t0, t1, dt)

#oscN = 3
#nH = 2
F = [3., 5, 10]
W = np.array([f*2*np.pi for f in F])
R = np.array([1, 2, 1])
Y = np.array([0, 0.5, np.pi])
K = np.array([[2.1, 3.2, -2.5, 1.2],
              [-1.2, 0.3, -6.1, -5.2],
              [7.9, 3.4, 0.1, 4.2],
             ])

params = np.column_stack((W, Y, R, K))


kursl = KurSL(params)
phase, amp, signals = kursl.generate(t)

# Precalculate Fourier transform parameters
freq = np.fft.fftfreq(len(t)-1, dt)
idx = np.r_[freq>=0] & np.r_[freq<20]

f, axes = plt.subplots(3, 2, figsize=(8,8))
for osc in range(len(W)):
    ax = axes[osc, 0]
    ax.plot(t[:-1], signals[osc], 'b')
    ax.plot(t[:-1], amp[osc], 'r')
    ax.plot(t[:-1], -amp[osc], 'r')

    if osc == 0:
        ax.set_title("Time series")
    elif osc == len(W)-1:
        ax.set_xlabel("Time [s]")

    # Power spectrum
    FT = np.fft.fft(signals[osc])
    ax = axes[osc, 1]
    ax.plot(freq[idx], np.abs(FT[idx]))

    if osc == 0:
        ax.set_title("Fourier spectrum")
    elif osc == len(W)-1:
        ax.set_xlabel("Frequency [Hz]")

plt.savefig("KurSL_example", dpi=100)
plt.show()
