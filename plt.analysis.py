#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 10})

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import subprocess

data=dict()
data=np.loadtxt("analysis.dat", skiprows=1)


plt.figure()
plt.plot(data[:,0], data[:,2], label="<Pee>")
plt.plot(data[:,0], data[:,3], label="<Peeb>")
plt.xlabel("Physical time")
#plt.yscale("log")
plt.legend()
plt.savefig("ana_serv.png")


plt.figure()
plt.plot(data[:,0], data[:,6], label="M0")
plt.xlabel("Physical time")
plt.legend()
plt.savefig("ana_M0.png")


