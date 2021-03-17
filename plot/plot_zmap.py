import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
import numpy as np
import os
import sys


if len(sys.argv) == 1:
    print("Usage ./plot_cmap.py [data file].\n")
    sys.exit(1)


def plotmap(fname):
    data = np.flip(np.loadtxt(fname), 1)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(np.transpose(data[:,:-1]), extent=[data[0,-1], data[-1,-1], -3,3], aspect='auto')   ## extent=[t0, tn, z0, z1]
    plt.ylabel("Z")
    plt.xlabel("Phy time")
    plt.colorbar(im)

    plt.savefig("writez.jpg")
    plt.close()

def plotmap1d(fname):
    data = np.loadtxt(fname)
    d = np.shape(data)[0]
    
    
    plt.figure(figsize=(8, 6))
    
    for i in range(0,d):
	plt.plot(data[i,1:], label=data[i,0], linewidth=1.0)
	
    #plt.ylim(-0.1, 0.1)
    
    plt.xlabel("Z")
    plt.legend()

    plt.savefig("writez1d.jpg")
    plt.close()

def plotmap1d_diff(fname):
    data = np.loadtxt(fname)
    d = np.shape(data)[0]
    
    
    plt.figure(figsize=(8, 6))
    
    for i in range(0,d):
	plt.plot(data[i,1:]-data[0,1:], label=data[i,0], linewidth=1.0)
	
    #plt.ylim(-0.1, 0.1)
    
    plt.xlabel("Z")
    plt.ylabel("ee(t) - ee(0)")
    plt.legend()

    plt.savefig("writez1d_diff.jpg")
    plt.close()

for fn in sys.argv[1:]:

    print("Plotting {}".format(fn))
    #plotmap(fn)
    plotmap1d(fn)
    plotmap1d_diff(fn)
