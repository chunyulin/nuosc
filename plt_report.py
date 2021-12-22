#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 10})

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import subprocess

cwd =  os.getcwd().split('/')[-1]
print(cwd)

p = subprocess.check_output(['ls -d alpha*'],shell=True)
runs = p.split()

data=dict()
for run in runs:
    data[run]=np.loadtxt("./{}/analysis.dat".format(run.decode()), skiprows=1)

da  = [1,2,3,4,5,6,7]
la  = ['max(|P|-1)','|M0|','Pee','Peeb','avgP','avgPb','ee-xx']
log = [1,           1,      0,   0,     0,     0,       0     ]

def compare(col, tag, log = 0):
    plt.figure()
    for k in data:
        plt.plot(data[k][:,0],data[k][:,col], label=k)
    #plt.plot(d2[:,0],(d2[:,col]-d2[0,col]), label=la[2])
    #plt.plot(d3[:,0],(d3[:,col]-d3[0,col]), label=la[3])
    #plt.plot(d4[:,0],(d4[:,col]-d4[0,col]), label=la[4])
    plt.xlabel("Physical time")
    plt.title("analysis {}".format(tag))
    if (log): plt.yscale("log")
    plt.legend()
    plt.savefig("ana_{}.png".format(tag))

NR,NC = 2,4
fig, ax = plt.subplots(nrows=NR, ncols=NC, figsize=(NC*5,NR*4), squeeze=False )
fig.suptitle("{}".format(cwd))

for i in range(len(da)):
    dcol = da[i]
    dlab = la[i]
    islog = log[i]
    axt = ax[int(i/NC), i%NC]

    for t in data:
        axt.plot(data[t][:,0],data[t][:,dcol], label=t)
        axt.set_xlabel("Time --  {}".format(dlab))
        if (islog): axt.set_yscale("log")
        axt.legend()

plt.savefig("repoty_{}.png".format(cwd))
