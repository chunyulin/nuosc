import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def plotrate():
    data1 = np.loadtxt("../rate_10.dat")
    data2 = np.loadtxt("../rate_50.dat")
    data3 = np.loadtxt("../rate_100.dat")

    plt.figure()
    plt.plot(data2[:,0],data1[:,10], label="v-integrated ||ex||, mu=10")
    plt.plot(data2[:,0],data2[:,10], label="v-integrated ||ex||, mu=50")
    plt.plot(data2[:,0],data3[:,10], label="v-integrated ||ex||, mu=100")
    plt.xlabel("Physical time")
    plt.yscale("log")
    plt.legend()
    plt.savefig("mu.jpg")


plotrate()
