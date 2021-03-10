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

for fn in sys.argv[1:]:

    print("Plotting {}".format(fn))
    plotmap(fn)
