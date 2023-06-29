#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 10})

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import subprocess

runs="*"
if len(sys.argv)>1:
   runs=sys.argv[1:]
print(runs)

#p = subprocess.check_output(['ls -d {}'.format(pattern) ],shell=True)
#runs = p.split()

data=dict()
for run in runs:
    data[run]=np.loadtxt("./{}/analysis.dat".format(run), skiprows=1)


def compare(col, tag, log = 0):
    plt.figure()
    for k in data:
        plt.plot(data[k][:,0],data[k][:,col], label=k, linewidth=1)
    #plt.plot(d2[:,0],(d2[:,col]-d2[0,col]), label=la[2])
    #plt.plot(d3[:,0],(d3[:,col]-d3[0,col]), label=la[3])
    #plt.plot(d4[:,0],(d4[:,col]-d4[0,col]), label=la[4])
    plt.xlabel("Physical time")
    plt.title("{}".format(tag))
    #plt.ylim(1e-18, 1e-13)
    if (log): plt.yscale("log")
    plt.legend()
    plt.savefig("ana_{}.png".format(tag))

#:maxrelP,    2:surv, survb,    4:avgP, avgPb,      6:aM0
compare(1,"max(P-1)", 1)
compare(2,"Pee")
compare(3,"Peeb")
compare(4,"avgP")
compare(5,"avgPb")
compare(6,"|M|", 0)
compare(7,"e-x", 1)
compare(8,"ELN_error", 1)
