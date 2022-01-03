#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})

import matplotlib.pyplot as plt
import numpy as np
import os, sys, subprocess

CWD =  os.getcwd().split('/')[-1]
files=sys.argv[1:]

## read grid configuation.
def readmeta(fn):
    print(fn)
    f = open(fn, 'r')
    nz, sz, dsz, _, _, dt = f.readline().split()
    nv, sv, dsv = f.readline().split()
    z_grid = [ float(x.strip()) for x in f.readline().split() ]
    v_grid = [ float(x.strip()) for x in f.readline().split() ]
    ##zgrid = np.fromfile(fn, dtype=np.double, count=sz)
    ##vgrid = np.fromfile(fn, dtype=np.double, count=sv)
    
    return int(sz), int(sv), float(dt), z_grid, v_grid
    
nz, nv, dt, z_grid, v_grid = readmeta("%s.meta"%files[0])
v, z = np.meshgrid(v_grid, z_grid)

NR,NC=4,1
VARS = ['P3']
nvar = len(VARS)

for fname in files:
    
    fn = fname
    print("Processing ", fn)
    
    with open(fn, 'rb') as f:
        data = np.fromfile(f, dtype=np.double, count=nz*nv*nvar) .reshape([nvar,nz,nv])
    
    time = dt*int(fname[3:9])

    fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*10,NR*2.5), squeeze=False, sharex=True, gridspec_kw = {'wspace':0.05, 'hspace':0.05} )
    fig.suptitle("T={:10.3f} :  {}/{}".format(time, CWD, fn), ha='right')
    idx=0

    ## P3(z,v)
    im = ax[idx,0].pcolormesh(z, v, data[0], cmap='RdBu_r', shading='auto', vmin=-1, vmax=1)
    ax[idx,0].set_ylabel(VARS[0])
    fig.colorbar(im, ax=ax[idx,0])
    ax[idx,0].set_xlabel("z")
    idx+=1

    ## v=1
    ax[idx,0].plot(z_grid, data[0,:,-1], linewidth=1)
    ax[idx,0].set_ylabel("P3 (v=1)")
    #ax[idx,0].set_ylim([-1,1])
    ax[idx,0].set_xticklabels([])
    idx+=1

    ## v=0
    ax[idx,0].plot(z_grid, data[0,:, (nv-1)>>1], linewidth=1)
    ax[idx,0].set_ylabel("P3 (v=0)")
    #ax[idx,0].set_ylim([-1,1])
    ax[idx,0].set_xticklabels([])
    idx+=1

    ## v=-1
    ax[idx,0].plot(z_grid, data[0,:,0], linewidth=1)
    ax[idx,0].set_ylabel("P3 (v=-1)")
    #ax[idx,0].set_ylim([-1,1])
    ax[idx,0].set_xticklabels([])
    idx+=1

    ## common 	
    #fig.tight_layout()
    plt.subplots_adjust(top=0.93)
    outname = "{}.png".format(os.path.splitext( os.path.basename(fn) )[0])
    plt.savefig(outname)
    plt.close()

