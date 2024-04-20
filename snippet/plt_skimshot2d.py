#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
import numpy as np
import os, sys, subprocess
import glob

#if len(sys.argv) == 1:
#    print("Usage ./plot_snapshot.py [data file].\n")
#    sys.exit(1)
#p = subprocess.check_output("ls {} -1".format(FILTER),shell=True)
#print(p)

CWD   = os.getcwd().split('/')[-1]
files = sys.argv[1:]
print("Input: ", files)

## read grid configuation.
def readmeta(fn):
    print(fn)
    f = open(fn, 'r')
    _, nz, _, sv = f.readline().split()
    z0, z1 = f.readline().split()
    z_grid = [ float(x.strip()) for x in f.readline().split() ]
    v_grid = [ float(x.strip()) for x in f.readline().split() ]
    ##zgrid = np.fromfile(fn, dtype=np.double, count=sz)
    ##vgrid = np.fromfile(fn, dtype=np.double, count=sv)
    dz = (float(z1)-float(z0))/int(nz)

    return int(nz), int(sv), z_grid, v_grid, dz

meta = glob.glob("*.meta")[0]
#nz, nv, z_grid, v_grid, dz = readmeta("%s.meta"%files[0])
nz, nv, z_grid, v_grid, dz = readmeta(meta)
v, z = np.meshgrid(v_grid, z_grid)

NR,NC=3,1
VARS = ['P1','P2','P3']
nvar = len(VARS)

for fname in files:
    
    fn = fname ##.decode()
    with open(fn, 'rb') as f:
        t    = np.fromfile(f, dtype=np.int32, count=1)[0]
        time = np.fromfile(f, dtype=np.double, count=1)[0]
        data = np.fromfile(f, dtype=np.double, count=nz*nv*nvar) .reshape([nvar,nz,nv])

    print("Processing ", fn, "T=", time)

    fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*12,NR*3), squeeze=False, sharex=True, gridspec_kw = {'wspace':0.05, 'hspace':0.05} )
    fig.suptitle("T= {:.2f}   dz= {}  src={}".format(time, dz, fn))
    #fig.suptitle("{}/{}".format(CWD, fn))
    im = []
    for nr in range(NR):
        for nc in range(NC):
            im_ = ax[nr,nc].pcolormesh(z, v, data[nr*NC+nc], cmap='RdBu_r', shading='auto')
            im.append(im_)
            ax[nr,nc].set_ylabel(VARS[nr*NC+nc])
            fig.colorbar(im_, ax=ax[nr,nc])
            if nr<=NR-1:
                ax[nr,nc].set_xticks([], [])

    for nc in range(NC):
        ax[NR-1, nc].set_xlabel("z")

    #fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    outname = "{}.png".format(os.path.splitext( os.path.basename(fn) )[0])
    plt.subplots_adjust(top=0.93)
    plt.savefig(outname)
    plt.close()

