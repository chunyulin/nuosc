#!/usr/bin/env python3
import os, sys, subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})

def readmeta(fn):
    f = open(fn, 'r')
    dt,sy,sz,sv = f.readline().split()
    ny,nz,nv    = f.readline().split()
    y0,y1,z0,z1 = f.readline().split()
    yc = [ float(x.strip()) for x in f.readline().split() ]
    zc = [ float(x.strip()) for x in f.readline().split() ]
    vyc = [ float(x.strip()) for x in f.readline().split() ]
    vzc = [ float(x.strip()) for x in f.readline().split() ]
    
    return float(dt), int(sy), int(sz), int(sv), yc, zc, vyc, vzc

def pltZY_P3(fname, zc, yc):

    CWD =  os.getcwd().split('/')[-1]
    NR,NC=3,1
    VARS = ['P3']
    nvar = len(VARS)

    with open(fname, 'rb') as f:
        t        = np.fromfile(f, dtype=np.int32, count=1)[0]
        phy_time = np.fromfile(f, dtype=np.double, count=1)[0]
        data     = np.fromfile(f, dtype=np.double, count=sy*sz*sv*nvar).reshape([nvar,sy,sz,sv])

    print("Processing {} at T={}".format(fname, phy_time))

    var=0
    vi=0
    z, y = np.meshgrid(zc, yc)    ## (h,v)

    ### Init plot
    fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*10,NR*3), squeeze=False, sharex=True, gridspec_kw = {'wspace':0, 'hspace':0.1} )
    fig.suptitle("P3  T={:10.3f}  step={}   {}".format(phy_time, t, CWD), ha='right')

    ## P3(z,v1)
    vi=-1
    im0 = ax[0,0].pcolormesh(z, y, data[var,:,:,vi], cmap='RdBu_r', shading='auto') #, vmin=-1, vmax=1)
    ax[0,0].set_ylabel("Y:   v=({},{})".format(vyc[vi],vzc[vi]))
    plt.colorbar(im0, ax=ax[0,0])
    
    ## P3(z,v2)
    vi=0
    im1 = ax[1,0].pcolormesh(z, y, data[var,:,:,vi], cmap='RdBu_r', shading='auto') #, vmin=-1, vmax=1)
    ax[1,0].set_ylabel("Y:   v=({},{})".format(vyc[vi],vzc[vi]))
    plt.colorbar(im1, ax=ax[1,0])
 
    ## P3(z,v3)
    vi=int(sv/2)
    im2 = ax[2,0].pcolormesh(z, y, data[var,:,:,vi], cmap='RdBu_r', shading='auto') #, vmin=-1, vmax=1)
    ax[2,0].set_ylabel("Y:   v=({},{})".format(vyc[vi],vzc[vi]))
    ax[2,0].set_xlabel("Z")
    plt.colorbar(im2, ax=ax[2,0])
   
    ## common 	
    #fig.tight_layout()
    plt.subplots_adjust(top=0.92, left=0.1, bottom=0.05)
    outname = "ZY{}.png".format(os.path.splitext( os.path.basename(fname) )[0])
    plt.savefig(outname)
    plt.close()
        
###### read grid configuation (2D).
files=sys.argv[1:]
dt, sy, sz, sv, yc, zc, vyc, vzc = readmeta("%s.meta"%files[0])
print("Meta data: dt= {} dim= {} x {} x {}".format( dt, sy, sz, sv) )

for fname in files:
    pltZY_P3(fname, zc, yc)
