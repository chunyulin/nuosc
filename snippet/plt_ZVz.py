#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
import numpy as np
import os, sys, subprocess
import glob

CWD =  os.getcwd().split('/')[-2:]
files=sys.argv[1:]

## read grid configuation.
def readmeta(fn):
    print(fn)
    f = open(fn, 'r')
    dt, nx, nz, sv = f.readline().split()
    x0, x1 = f.readline().split()
    z0, z1 = f.readline().split()
    x_grid = [ float(x.strip()) for x in f.readline().split() ]
    z_grid = [ float(x.strip()) for x in f.readline().split() ]
    vx_grid = [ float(x.strip()) for x in f.readline().split() ]
    vz_grid = [ float(x.strip()) for x in f.readline().split() ]
    dz = (float(z1)-float(z0))/int(nz)

    return int(nx), int(nz), int(sv), x_grid, z_grid, vx_grid, vz_grid, dz

meta = glob.glob("*.meta")[0]
nx, nz, nv, xc, zc, vxc, vzc, dz = readmeta(meta)
vz, z = np.meshgrid(vzc, zc)   ## return nzxnz
print(vz.shape)
print(z.shape)


NR,NC=3,1
VARS = ['P3']
nvar = len(VARS)

for fname in files:
    
    fn = fname ##.decode()
    with open(fn, 'rb') as f:
        t    = np.fromfile(f, dtype=np.int32, count=1)[0]
        time = np.fromfile(f, dtype=np.double, count=1)[0]
        data = np.fromfile(f, dtype=np.double, count=nx*nz*nv*nvar) .reshape([nvar,nx,nz,nv])

    print("Processing ", fn, "T=", time)

    fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*13,NR*3), squeeze=False, sharex=True, gridspec_kw = {'wspace':0.05, 'hspace':0.05} )

    ## VAR[0] ==============
    for x in [0,1,2]:
        row = x
        col = 0
  
        var=0
        im = ax[row,col].pcolormesh(z, vz, data[var,x,:,:], cmap='RdBu_r', shading='auto')
        fig.colorbar(im, ax=ax[row,col])
        ax[row,col].set_aspect("auto")

        ax[row,col].set_xlabel("Z")
        ax[row,col].set_ylabel("Vz  @ x={}".format(xc[x]))
        if (x==0):
            ax[row,col].set_title("{}, T={:10.3f}, step={}    {}/{}".format(VARS[var], time, t, CWD[0],CWD[1]), loc='left')
        if (x!=2):
            ax[row,col].set_xticks([],[])
            ax[row,col].set_xlabel("")

    #fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    outname = "VZ{}.png".format(os.path.splitext( os.path.basename(fn) )[0])
    plt.subplots_adjust(top=0.93)
    plt.savefig(outname)
    plt.close()

