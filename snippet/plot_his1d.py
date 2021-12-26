import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def monitor_conserved(tag, fname):
    data = np.loadtxt(fname)
    
    #plt.figure()
    #plt.plot(data[:,0],data[:,3], label="std: |ex|")
    #plt.plot(data[:,0],1-data[:,6], label="std: |1-bee|")
    #plt.fill_between(data[:,0],   data[:,1],  data[:,2], alpha=0.1)
    #plt.fill_between(data[:,0], 1-data[:,4],1-data[:,5], alpha=0.1)
    #plt.yscale("log")
    #plt.xlabel("Physical time")
    #plt.legend()
    #plt.savefig("hist.jpg")

    plt.figure()
    plt.plot(data[:,0],np.abs(data[:,1]-data[0,1]), label="avgP")
    plt.plot(data[:,0],np.abs(data[:,2]-data[0,2]), label="avgPb")
    plt.plot(data[:,0],np.abs(data[:,3]-data[0,3]), label="amaxrelP")
    plt.plot(data[:,0],np.abs(data[:,4]-data[0,4]), label="amaxrelN")
    plt.plot(data[:,0],np.abs(data[:,5]-data[0,5]), label="aM0")
    plt.plot(data[:,0],np.abs(data[:,6]-data[0,6]), label="aM1")
    plt.xlabel("Physical time")
    plt.yscale("log")
    plt.legend()
    plt.savefig("{}.png".format(tag))

def his1d(tag, fname, col):
    data = np.loadtxt(fname)

    plt.figure()
    plt.plot(data[:,0],data[:,col], label=tag)

    plt.xlabel("Physical time")
    plt.legend()
    plt.savefig("{}.png".format(tag))

fname = sys.argv[1]

monitor_conserved("conserved", fname)

#his1d("avgPb", dat, 2)
#plot1d("amaxrelP", dat, 3)
#plot1d("amaxrelN", dat, 4)
#plot1d("aM0", dat, 5)
#plot1d("aM1", dat, 6)
