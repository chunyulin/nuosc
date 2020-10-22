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
    data = np.loadtxt(fname)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(data[:,1:], extent=[-1,1, data[-1,0], data[0,0]], aspect='auto')
    plt.xlabel("Z")
    plt.ylabel("Phy time")
    plt.colorbar(im)

    plt.savefig("writez.jpg")
    plt.close()

for fn in sys.argv[1:]:

    print("Plotting {}".format(fn))
    plotmap(fn)
