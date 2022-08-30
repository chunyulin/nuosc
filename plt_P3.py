#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})

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
    _, nz, _, sv = f.readline().split()
    z0, z1 = f.readline().split()
    z_grid = [ float(x.strip()) for x in f.readline().split() ]
    v_grid = [ float(x.strip()) for x in f.readline().split() ]
    ##zgrid = np.fromfile(fn, dtype=np.double, count=sz)
    ##vgrid = np.fromfile(fn, dtype=np.double, count=sv)
    dz = (float(z1)-float(z0))/int(nz)

    return int(nz), int(sv), z_grid, v_grid, dz

    
#nz, nv, dt, z_grid, v_grid = readmeta("%s.meta"%files[0])
meta = glob.glob("*.meta")[0]
nz, nv, z_grid, v_grid, dz = readmeta(meta)

v, z = np.meshgrid(v_grid, z_grid)

NR,NC=1,1
VARS = ['P1','P2','P3']
nvar = len(VARS)

for fname in files:
    
    fn = fname
    print("Processing ", fn)
    
    with open(fn, 'rb') as f:
        t    = np.fromfile(f, dtype=np.int32, count=1)[0]
        time = np.fromfile(f, dtype=np.double, count=1)[0]
        data = np.fromfile(f, dtype=np.double, count=nz*nv*nvar).reshape([nvar,nz,nv])

    fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*12,NR*4), squeeze=False, sharex=True,  gridspec_kw = {'wspace':0.05, 'hspace':0.1} )
    #fig.suptitle("T={:10.3f} :  {}/{}".format(time, CWD[0],CWD[1]), ha='right')
    #fig.suptitle("T= {:.2f}   dz= {}  src={}".format(time, dz, fn))

    var = 2
    idx=0

    ## P3(z,v)
    #im = ax[idx,0].pcolormesh(z, v, data[var], cmap='RdBu_r', shading='auto', vmin=-1, vmax=1)
    im = ax[idx,0].pcolormesh(z, v, data[var], cmap='RdBu_r', shading='auto')
    ax[idx,0].set_ylabel("v")
    fig.colorbar(im, ax=ax[idx,0])
    ax[idx,0].set_xlabel("z")
    ax[idx,0].set_title("{}, T={:10.3f}, step={}    {}/{}".format(VARS[var], time, t, CWD[0],CWD[1]), loc='left')

    ## common 	
    #fig.tight_layout()
    plt.subplots_adjust(top=0.93)
    outname = "{}.png".format(os.path.splitext( os.path.basename(fn) )[0])
    plt.savefig(outname)
    plt.close()

