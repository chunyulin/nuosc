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
    with open(fname, 'rb') as f:
	time, vz0, vz1, z0, z1 = np.fromfile(f, dtype=np.double, count=5)
        #f.seek(8, os.SEEK_SET)
        nz,nvz,gz = np.fromfile(f, dtype=np.intc, count=3)

        data   = np.fromfile(f, dtype=np.double, count=8*(nz+2*gz)*nvz).reshape([8, nvz, nz+2*gz])

    dz = 1.0/nz
    z_grid = np.linspace(z0-1.5*dz, z1+1.5*dz, nz+2*gz)
    v_grid = np.linspace(vz0,vz1,nvz)
    z, v = np.meshgrid(z_grid, v_grid)

    #print (z.shape)
    #print (v.shape)
    #print (vee.shape)
    NR=2
    NC=4
    fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*4.5,NR*3) )
    fig.suptitle("Physical time:{}     dz={}     (nv,nz)=({} {})      src file={}".format(time, dz, nvz, nz, fname))
    li = ['ee','exr','exi','bee','bexr','bexi']
    im = []
    for nr in range(NR):
	for nc in range(NC):
	    im_ = ax[nr,nc].pcolormesh(z, v, data[nr*NC+nc], cmap='RdBu_r')
    	    im.append(im_)
    	    ax[nr,nc].set_ylabel(li[nr*NC+nc])
	    fig.colorbar(im_, ax=ax[nr,nc])
    
    for nc in range(NC):
	ax[NR-1, nc].set_xlabel("z")

    #fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    outname = "{}.jpg".format(os.path.splitext( os.path.basename(fname) )[0])
    plt.subplots_adjust(top=0.93)
    plt.savefig(outname)
    plt.close()

for fn in sys.argv[1:]:
    print("Plotting {}".format(fn))
    plotmap(fn)
