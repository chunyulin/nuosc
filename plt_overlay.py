#!/usr/bin/env python3
import os, sys, subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'lines.linewidth': 0.5})
from mpl_toolkits.axes_grid1 import make_axes_locatable

def readmeta(fn):
    print(fn)
    f = open(fn, 'r')
    dt,nx,nz,sv = f.readline().split()
    x0,x1 = f.readline().split()
    z0,z1 = f.readline().split()
    xc  = [ float(x.strip()) for x in f.readline().split() ]
    zc  = [ float(x.strip()) for x in f.readline().split() ]
    vxc = [ float(x.strip()) for x in f.readline().split() ]
    vyc = [ float(x.strip()) for x in f.readline().split() ]
    vzc = [ float(x.strip()) for x in f.readline().split() ]
    return float(dt), int(nx), int(nz), int(sv), xc, zc, vxc, vyc, vzc

###### read grid configuation (2D).
files=sys.argv[1:]

meta = subprocess.check_output(['ls -1 *.meta | head -1'],shell=True).decode()[:-1]
dt, nx, nz, sv, xc, zc, vxc, vyc, vzc = readmeta(meta)
print("Meta data: dt= {} dim= {} x {} x {}".format( dt, nx, nz, sv) )

### Init plot
sxc = [nx//2] ##### X coord index to be plot

NR,NC=len(sxc),1
fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*8,NR*3), squeeze=False, sharex=True, gridspec_kw = {'wspace':0, 'hspace':0.001})
axi = 0
var=0
yi=0
v = 0

for xi in range(len(sxc)):
  xx = np.ones( len(zc) )
  ax[xi,axi].plot(zc, 0.8*xx, "k+-",  markersize=2) ##, markerfacecolor="black",  markeredgecolor="black")

  for fname in files:
    print(fname)
    CWD =  os.getcwd().split('/')[-1]
    VARS = ['EE']
    nvar = len(VARS)
    
    with open(fname, 'rb') as f:
        t        = np.fromfile(f, dtype=np.int32, count=1)[0]
        phy_time = np.fromfile(f, dtype=np.double, count=1)[0]
        data     = np.fromfile(f, dtype=np.double, count=nx*nz*sv*nvar).reshape([nvar,nx,nz,sv])

    print("Processing {} at T={}".format(fname, phy_time))

    ax[xi,axi].set_xlabel("Z")
    ax[xi,axi].plot(zc, data[var,xi,:,v].flatten(), label="t= {:.2f}".format(phy_time) )

    ax[xi,axi].set_ylabel(f"v=({vxc[v]:.2f} {vyc[v]:.2f} {vzc[v]:.2f}) x={xc[xi]:.2f}")
    #ax[xi,axi].set_xlim([ zc[0]/2,zc[-1]/2 ])


plt.legend()
plt.savefig("adv.png",  dpi=300, bbox_inches='tight')
plt.close()

