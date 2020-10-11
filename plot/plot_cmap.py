import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18})

import matplotlib.pyplot as plt
import numpy as np
import os

fname="../stat_0009.bin"

with open(fname, 'rb') as f:
    time = np.fromfile(f, dtype=np.double, count=1)
    #f.seek(8, os.SEEK_SET)
    nz,nvz,gz = np.fromfile(f, dtype=np.intc, count=3)

    data   = np.fromfile(f, dtype=np.double, count=6*(nz+2*gz)*nvz).reshape([6, nvz, nz+2*gz])

dz = 1.0/nz
z_grid = np.linspace(-1.5*dz, 1+1.5*dz, nz+2*gz)
v_grid = np.linspace(0,1,nvz)
z, v = np.meshgrid(z_grid, v_grid)

def plotmap(d, fname):
    #print (z.shape)
    #print (v.shape)
    #print (vee.shape)
    NR=2
    NC=3
    fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*8,NR*5) )
    li = ['ee','exr','exi','bee','bexr','bexi']
    im = []
    for nr in range(NR):
	for nc in range(NC):
	    im_ = ax[nr,nc].pcolormesh(z, v, d[nr*NC+nc], cmap='RdBu_r')
    	    im.append(im_)
    	    ax[nr,nc].set_ylabel(li[nr*NC+nc])
	    fig.colorbar(im_, ax=ax[nr,nc])
    
    for nc in range(NC):
	ax[NR-1, nc].set_xlabel("z")

    ax[0,0].set_title("Physical time: {}".format(time))
    
    #fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    plt.savefig(fname)
    plt.close()


plotmap(data, "cmap_0009.jpg")
