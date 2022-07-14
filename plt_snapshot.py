#!/usr/bin/env python3
import os, sys, subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'lines.linewidth': 1})
from mpl_toolkits.axes_grid1 import make_axes_locatable

def readmeta(fn):
    print(fn)
    f = open(fn, 'r')
    dt,nz,np,nv = f.readline().split()
    z0,z1 = f.readline().split()
    zc = [ float(x.strip()) for x in f.readline().split() ]
    vc = [ float(x.strip()) for x in f.readline().split() ]
    return float(dt), int(nz), int(np), int(nv), zc, vc

def plot_snapshot(fname, zc, vc): 

    CWD =  os.getcwd().split('/')[-1]

    NR,NC=1,1
    VARS = ['EE']
    nvar = len(VARS)
    
    with open(fname, 'rb') as f:
        t        = np.fromfile(f, dtype=np.int32, count=1)[0]
        phy_time = np.fromfile(f, dtype=np.double, count=1)[0]
        data     = np.fromfile(f, dtype=np.double, count=sz*sp*sv*nvar).reshape([nvar,sz,sp,sv])

    print("Processing {} at T={}".format(fname, phy_time))
    
    var=0
    yi=0
    
    ### Init plot
    fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*10,NR*3), squeeze=False, sharex=True, gridspec_kw = {'wspace':0, 'hspace':0.1} )
    axi = 0
    
    ## P(z,v+)
    v = 0

    ax[axi,0].set_title("                                Rho_EE, T={:10.3f}, step={}, {}".format(phy_time, t, CWD), loc='left')
    ax[axi,0].set_xlabel("Z")
    ax[axi,0].plot(zc, data[var,:,:,v].flatten(), linewidth=1)
    if (phy_time<zc[-1]): ax[axi,0].axvline(x=vc[v]*phy_time, c=ax[axi,0].get_lines()[-1].get_color())

    ax[axi,0].set_ylabel("v={}".format(vc[v]))
    #ax[axi,0].set_ylim([-1,1])
    axi+=1


    outname = "{}.png".format(os.path.splitext( os.path.basename(fname) )[0])
    plt.savefig(outname)
    plt.close()

###### read grid configuation (2D).
files=sys.argv[1:]

meta = subprocess.check_output(['ls -1 *.meta | head -1'],shell=True).decode()[:-1]
dt, sz, sp, sv, zc, vc = readmeta(meta)
print("Meta data: dt= {} dim= {} x {} x {}".format( dt, sz, sp, sv) )    
for fname in files:
    print(fname)
    plot_snapshot(fname, zc, vc)
