#!/usr/bin/env python3
import os, sys, subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'lines.linewidth': 1})
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
./cpltZVz_P3.py <files to compare>
"""
import argparse
    
def readmeta(fn):
    print("Reading meta data: %s" % fn)
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
parser = argparse.ArgumentParser()
parser.add_argument('runs', nargs='*', default=[])
parser.add_argument('-l', '--list', nargs='+')
args = parser.parse_args()
if len(args.runs) ==0:
    p = subprocess.check_output(['ls -d */'],shell=True)
    args.runs = [x.decode()[:-1] for x in p.split() ]
#print(args.runs)
#print(args.list)

## get binary file list from the first folder
p = subprocess.check_output(['(cd {}; ls -d *.bin)'.format(args.runs[0])],shell=True)
dfns = [x.decode() for x in p.split() ]

VARS = ['P3']
NR,NC=3,1
nvar=len(VARS)

meta = dict()
# get meta data
for runset in args.runs:
    #dt, sy, sz, sv, yc, zc, vyc, vzc = readmeta("{}/{}.meta".format(runset,files[0]) )
    meta[runset] = readmeta("{}/{}.meta".format(runset,dfns[0]) )
    
for dfn in dfns:

    print("Processing %s ..."%dfn)

    ### Init plot
    fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*10,NR*3), squeeze=False, sharex=True, gridspec_kw = {'wspace':0, 'hspace':0.1} )
    yi=0
    var=0
    
    for run in args.runs:
            
        sy=meta[run][1]
        sz=meta[run][2]
        sv=meta[run][3]
        zc=meta[run][5]
        vc=meta[run][7]
        
        with open("{}/{}".format(run,dfn), 'rb') as f:
            t        = np.fromfile(f, dtype=np.int32, count=1)[0]
            phy_time = np.fromfile(f, dtype=np.double, count=1)[0]
            data     = np.fromfile(f, dtype=np.double, count=sy*sz*sv*nvar).reshape([nvar,sy,sz,sv])

        ## P(z,v+)
        v1=-1
        ax[0,0].set_title("P3:   T={:10.3f}    step={}".format(phy_time, t), ha='right')
        ax[0,0].plot(zc, data[var,yi,:, v1])
        ax[0,0].set_ylabel("v={}".format(vc[v1]))
        v1=int(sv/2)
        ax[1,0].plot(zc, data[var,yi,:, v1])
        ax[1,0].set_ylabel("v={}".format(vc[v1]))
        v1=0
        ax[2,0].plot(zc, data[var,yi,:, 0], label=run)
        ax[2,0].set_ylabel("v={}".format(vc[v1]))
        ax[2,0].set_xlabel("Z")

    
    plt.legend()
    outname = "c{}.png".format( os.path.splitext( os.path.basename(dfn) )[0] )
    plt.savefig(outname)
    plt.close()


    
#for fname in files:
#    plot_ZVz_P3(fname, zc, vzc)  