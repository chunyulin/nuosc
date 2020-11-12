import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def plotrate(fname):
    data = np.loadtxt(fname)
    
    plt.figure()
    plt.plot(data[:,0],data[:,3], label="std: |ex|")
    plt.plot(data[:,0],1-data[:,6], label="std: |1-bee|")
    plt.fill_between(data[:,0],   data[:,1],  data[:,2], alpha=0.1)
    plt.fill_between(data[:,0], 1-data[:,4],1-data[:,5], alpha=0.1)
    #plt.yscale("log")
    plt.xlabel("Physical time")
    plt.legend()
    plt.savefig("hist.jpg")

    plt.figure()
    plt.plot(data[:,0],data[:,7], label="|1-bee|")
    plt.xlabel("Physical time")
    plt.legend()
    plt.savefig("rhoxx.jpg")


plotrate("../rate.dat")
