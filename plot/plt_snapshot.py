#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
import numpy as np
import os
import sys



if len(sys.argv) == 1:
    print("Usage ./plot_snapshot.py [data file].\n")
    sys.exit(1)


def plotmap(fname):
    v0,v1 = -1,1
    NR=3
    NC=1

    with open(fname, 'rb') as f:
        time, z0, z1 = np.fromfile(f, dtype=np.double, count=3)
        nz,nvz,gz    = np.fromfile(f, dtype=np.intc, count=3)
        #f.seek(8, os.SEEK_SET)

        data   = np.fromfile(f, dtype=np.double, count=NR*NC*(nz+2*gz)*nvz).reshape([NR*NC, nz+2*gz, nvz])

    dz = (z1-z0)/nz
    dv = 2.0/nvz
    z_grid = np.linspace(z0-(gz-0.5)*dz, z1+(gz-0.5)*dz, nz+2*gz)
    #v_grid = np.linspace(v0+0.5*dv, v1-0.5*dv, nvz)
    v_grid = np.linspace(-1,1, nvz)
    v, z = np.meshgrid(v_grid, z_grid)

    #print (z.shape)
    #print (v.shape)
    #print (vee.shape)
    fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*9,NR*2), squeeze=False )
    fig.suptitle("Physical time:{:.2f}   dz={}   (nz,nv)=({} {})      src file={}".format(time, dz, nz, nvz, fname))
    li = ['P1','P2','P3','bee','bexr','bexi']
    im = []
    for nr in range(NR):
        for nc in range(NC):
            im_ = ax[nr,nc].pcolormesh(z, v, data[nr*NC+nc], cmap='RdBu_r', shading='auto')
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
