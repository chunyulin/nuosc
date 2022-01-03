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
    dt,sy,sz,sv = f.readline().split()
    ny,nz,nv    = f.readline().split()
    y0,y1,z0,z1 = f.readline().split()
    yc = [ float(x.strip()) for x in f.readline().split() ]
    zc = [ float(x.strip()) for x in f.readline().split() ]
    vyc = [ float(x.strip()) for x in f.readline().split() ]
    vzc = [ float(x.strip()) for x in f.readline().split() ]    
    return float(dt), int(sy), int(sz), int(sv), yc, zc, vyc, vzc

def plot_ZVz_P3(fname, zc, vc): 

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
    yi=0
    v, z = np.meshgrid(vc, zc)    ## (v,h)
    
    ### Init plot
    fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*10,NR*3), squeeze=False, sharex=True, gridspec_kw = {'wspace':0, 'hspace':0.1} )
    #fig.suptitle("P3  T={:10.3f} ({})    ".format(phy_time, t), ha='right')

    ## P3(z,v)
    axi=0
    ax[axi,0].set_title("P3, T={:10.3f}, step={}, {}".format(phy_time, t, CWD), loc='left')
    #contours = ax[axi,0].contour(z, v, data[var,yi,:,:], 3, colors='black')
    #ax[axi,0].clabel(contours, inline=True, fontsize=6)
    im = ax[axi,0].pcolormesh(z, v, data[var,yi,:,:], cmap='RdBu_r', shading='auto') #, vmin=-1, vmax=1)
    
    ax[axi,0].set_ylabel("vz")
    axi+=1

    ## P(z,v+)
    v1,v2=-1,0

    ax[axi,0].plot(zc, data[var,yi,:,v1], linewidth=1)
    if (phy_time<zc[-1]): ax[axi,0].axvline(x=vc[v1]*phy_time, c=ax[axi,0].get_lines()[-1].get_color())

    ax[axi,0].plot(zc, data[var,yi,:,v2], linestyle='--', linewidth=1)
    if (phy_time<zc[-1]): ax[axi,0].axvline(x=vc[v2]*phy_time, c=ax[axi,0].get_lines()[-1].get_color())

    ax[axi,0].set_ylabel("v={} {} ".format(vc[v1], vc[v2]))
    #ax[axi,0].set_ylim([-1,1])
    axi+=1

    ## P(z,v0)
    vi=int(sv/2)
    ax[axi,0].plot(zc, data[var,yi,:,vi], linewidth=1)
    ax[axi,0].set_ylabel("v={}".format(vc[vi]))
    #ax[axi,0].set_ylim([-1,1])
    ax[axi,0].set_xlabel("z")

    ## common 	
    #fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    pos1 = ax[0,0].get_position()
    cbar_ax = fig.add_axes([pos1.x1 + 0.01, pos1.y0, 0.03, pos1.y1-pos1.y0])
    fig.colorbar(im, cax=cbar_ax)    
   
    outname = "{}.png".format(os.path.splitext( os.path.basename(fname) )[0])
    plt.savefig(outname)
    plt.close()

###### read grid configuation (2D).
files=sys.argv[1:]
dt, sy, sz, sv, yc, zc, vyc, vzc = readmeta("%s.meta"%files[0])
print("Meta data: dt= {} dim= {} x {} x {}".format( dt, sy, sz, sv) )    
    
for fname in files:
    plot_ZVz_P3(fname, zc, vzc)    
