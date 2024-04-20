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
    dt,nz,np,nv = f.readline().split()
    z0,z1 = f.readline().split()
    zc = [ float(x.strip()) for x in f.readline().split() ]
    vc = [ float(x.strip()) for x in f.readline().split() ]
    return float(dt), int(nz), int(np), int(nv), zc, vc

###### read grid configuation (2D).
files=sys.argv[1:]

meta = subprocess.check_output(['ls -1 *.meta | head -1'],shell=True).decode()[:-1]
dt, sz, sp, sv, zc, vc = readmeta(meta)
print("Meta data: dt= {} dim= {} x {} x {}".format( dt, sz, sp, sv) )    


### Init plot
NR,NC=1,1
fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*7,NR*3), squeeze=False, sharex=True, gridspec_kw = {'wspace':0, 'hspace':0.1})
axi = 0
var=0
yi=0
v = 0

xx = np.ones( len(zc[::sp]) )
ax[axi,0].plot(zc[::sp], 0.8*xx, "k+-",  markersize=2) ##, markerfacecolor="black",  markeredgecolor="black")


for fname in files:
    print(fname)

    CWD =  os.getcwd().split('/')[-1]

    VARS = ['EE']
    nvar = len(VARS)
    
    with open(fname, 'rb') as f:
        t        = np.fromfile(f, dtype=np.int32, count=1)[0]
        phy_time = np.fromfile(f, dtype=np.double, count=1)[0]
        data     = np.fromfile(f, dtype=np.double, count=sz*sp*sv*nvar).reshape([nvar,sz,sp,sv])

    print("Processing {} at T={}".format(fname, phy_time))
    
    ax[axi,0].set_xlabel("Z")
    ax[axi,0].plot(zc, data[var,:,:,v].flatten(), label="t= {:.2f}".format(phy_time) )

    ax[axi,0].set_ylabel("v={}".format(vc[v]))
    #ax[axi,0].set_xlim([ zc[0]/2,zc[-1]/2 ])

plt.legend()
plt.savefig("adv.png",  dpi=300, bbox_inches='tight')
plt.close()

