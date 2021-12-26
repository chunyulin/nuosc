#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 10})

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import subprocess

p = subprocess.check_output(['ls G0_*.dat'],shell=True)
runs = p.split()

data=dict()
for run in runs:
    data[run]=np.loadtxt("{}".format(run.decode()), skiprows=0)

plt.figure()
for run in runs:
    plt.plot(data[run][:,0],data[run][:,1], label=run)
plt.axhline(y=0, color='k', linestyle='--')

plt.xlabel("v")
plt.ylabel("Gnu")
#plt.yscale("log")
plt.legend()
plt.xlim([-1,1])
plt.ylim([-0.8,0.2])
plt.grid()
plt.savefig("G0.png")
