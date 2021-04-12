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
    basename = os.path.basename(fname).split('.')[0]
    with open(fname) as f:
	nz, z0, z1 = np.array( f.readline().split(' ')).astype(np.float)

    print("Plotting {} in z [{}:{}] ({})".format(fn, z0, z1, nz))

    plt.figure(figsize=(8, 6))

    data = np.loadtxt(fname, skiprows=1)
    im = plt.imshow(data[:,1:], extent=[float(z0), float(z1), data[0,0], data[-1,0]], origin='lower', aspect='auto', interpolation='none')
    plt.xlabel("Z")
    plt.ylabel("Phy time")
    plt.title(basename)
    plt.colorbar(im)

    plt.plot([z0,0,z1],[np.abs(z0), 0, np.abs(z1)], 'r--')


    plt.savefig("{}.jpg".format(basename))
    plt.close()
    
def plot1dmap(fname):
    basename = os.path.basename(fname).split('.')[0]
    with open(fname) as f:
	nz, z0, z1 = np.array( f.readline().split(' ')).astype(np.float)
    dz=(z1-z0)/nz
    Z = np.linspace(z0+0.5*dz,z1-0.5*dz,nz)

    print("Plotting {} in z [{}:{}] ({})".format(fn, z0, z1, nz))
    data = np.loadtxt(fname, skiprows=1)
    nt    = np.shape(data)[0]
    pltnt = range(1,nt, nt//5)
    
    ## 1d
    plt.figure(figsize=(8, 6))
    for i in [0]+pltnt:
        plt.plot(Z, data[i,1:], label=data[i,0], linewidth=0.9)
    plt.xlabel("Z")
    plt.ylabel(basename)
    lg=plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("{}_1d.jpg".format(basename), bbox_inches='tight', bbox_extra_artists=(lg,))
    plt.close()

    ## 1d diff
    plt.figure(figsize=(8, 6))
    for i in pltnt:
        plt.plot(Z, data[i,1:]-data[i-1,1:], label=data[i,0], linewidth=.9)
    plt.xlabel("Z")
    plt.ylabel("{}, diff from the last time".format(basename))
    lg=plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("{}_1d_diff.jpg".format(basename), bbox_inches='tight', bbox_extra_artists=(lg,))
    plt.close()
    
    
for fn in sys.argv[1:]:
    plotmap(fn)
    plot1dmap(fn)
