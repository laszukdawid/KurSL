#!/usr/bin/python
# encoding: UTF-8

import logging
import numpy as np
import pylab as plt
import sys

from kursl import KurslMethod

# Import signal
S = np.loadtxt("examples/epilepsy_signal.txt")
fs = 173.61 # Hz
t = np.linspace(0, S.size/fs, S.size)

# Set verbose logging
logger = logging.getLogger(__file__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Define KurslMethod
kursl = KurslMethod(nH=2, max_osc=4,
                    nwalkers=60, niter=100,
                    energy_ratio=0.4)
theta = kursl.run(t, S)

_, _, s_rec = kursl.model(t, theta)
signal = np.sum(s_rec, axis=0)

plt.plot(t, S)
plt.plot(t[:-1], signal)
plt.show()
