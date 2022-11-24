#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
import numpy as np
import os, sys, subprocess

d = np.fromfile("test_fft.dat")

plt.figure()
plt.plot(d[:,0])
plt.plot(d[:,0])
plt.plot(sqrt(d[:,2]**2+d[:,3]**2))
plt.savefig("test_fft.png")
