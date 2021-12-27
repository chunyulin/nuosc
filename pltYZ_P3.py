#!/usr/bin/env python3
import os, sys, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'lines.linewidth': 1})
import matplotlib.pyplot as plt

CWD =  os.getcwd().split('/')[-1]
files=sys.argv[1:]

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
    
###### read grid configuation (2D).
dt, sy, sz, sv, yc, zc, vyc, vzc = readmeta("%s.meta"%files[0])
z, y = np.meshgrid(zc, yc)    ## (h,v)
print("Meta data: dt={} dim={}x{}x{}".format( dt, sy, sz, sv) )


NR,NC=1,1
VARS = ['P3']
nvar = len(VARS)

for fname in files:
    
    with open(fname, 'rb') as f:
        t        = np.fromfile(f, dtype=np.int32, count=1)[0]
        phy_time = np.fromfile(f, dtype=np.double, count=1)[0]
        data     = np.fromfile(f, dtype=np.double, count=sy*sz*sv*nvar).reshape([nvar,sy,sz,sv])

    print("Processing {} at T={}".format(fname, phy_time))

    var=0
    vi=0
    
    plt.figure()
    plt.title("T={:3f} ({})     {}".format(phy_time, t, fname) )
    contours = plt.contour(z, y, data[var,:,:,vi], 2, colors='black')
    #plt.clabel(contours, inline=True, fontsize=8)
    im = plt.pcolormesh(z, y, data[var,:,:,vi], cmap='RdBu_r', alpha=0.7, shading='auto') ##, vmin=-1, vmax=1)
    plt.xlabel("Z:   v=({},{})".format(vyc[vi],vzc[vi]));
    plt.colorbar();


    ## common 	
    #fig.tight_layout()
    #plt.subplots_adjust(top=0.93)
    outname = "{}.png".format(os.path.splitext( os.path.basename(fname) )[0])
    plt.savefig(outname)
    plt.close()

