#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})

import matplotlib.pyplot as plt
import numpy as np
import os, sys, subprocess

#if len(sys.argv) == 1:
#    print("Usage ./plot_snapshot.py [data file].\n")
#    sys.exit(1)
#p = subprocess.check_output("ls {} -1".format(FILTER),shell=True)
#print(p)


CWD =  os.getcwd().split('/')[-1]
files=sys.argv[1:]


## read grid configuation.
def readmeta(fn):
    print(fn)
    f = open(fn, 'r')
    nz, sz, dsz, _, _ = f.readline().split()
    nv, sv, dsv = f.readline().split()
    z_grid = [ float(x.strip()) for x in f.readline().split() ]
    v_grid = [ float(x.strip()) for x in f.readline().split() ]
    ##zgrid = np.fromfile(fn, dtype=np.double, count=sz)
    ##vgrid = np.fromfile(fn, dtype=np.double, count=sv)
    
    return int(sz), int(sv), z_grid, v_grid
    
nz, nv, z_grid, v_grid = readmeta("%s.meta"%files[0])
v, z = np.meshgrid(v_grid, z_grid)

NR,NC=3,1
VARS = ['P1','P2','P3']
nvar = len(VARS)


for fname in files:
    
    fn = fname ##.decode()
    print("Processing ", fn)
    
    with open(fn, 'rb') as f:
        data = np.fromfile(f, dtype=np.double, count=nz*nv*nvar) .reshape([nvar,nz,nv])

    fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*10,NR*3), squeeze=False, sharex=True, gridspec_kw = {'wspace':0.05, 'hspace':0.05} )
    #fig.suptitle("Physical time:{:.2f}   dz={}   (nz,nv)=({} {})      src file={}".format(time, dz, nz, nvz, fname))
    fig.suptitle("v = {}, {}/{}".format(v_grid[-1], CWD, fn))
    for nr in range(NR):
        for nc in range(NC):
            ax[nr,nc].plot(z_grid, data[nr*NC+nc,:,-1], linewidth=1)
            ax[nr,nc].set_ylabel(VARS[nr*NC+nc])
            if nr==NR-1:
                ax[nr,nc].set_xticklabels([])

    for nc in range(NC):
        ax[NR-1, nc].set_xlabel("z")

    #fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    outname = "{}v{}.png".format(os.path.splitext( os.path.basename(fn) )[0], v_grid[-1])
    plt.subplots_adjust(top=0.93)
    plt.savefig(outname)
    plt.close()

