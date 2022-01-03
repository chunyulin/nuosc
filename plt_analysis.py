#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 10})

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import subprocess

p = subprocess.check_output(['ls -d */'],shell=True)
runs = p.split()

data=dict()
for run in runs:
    data[run]=np.loadtxt("./{}/analysis.dat".format(run.decode()), skiprows=1)
    #except:
    #    continue


def compare(col, tag, log = 0):
    plt.figure()
    for k in data:
        plt.plot(data[k][:,0],data[k][:,col], label=k.decode())
    #plt.plot(d2[:,0],(d2[:,col]-d2[0,col]), label=la[2])
    #plt.plot(d3[:,0],(d3[:,col]-d3[0,col]), label=la[3])
    #plt.plot(d4[:,0],(d4[:,col]-d4[0,col]), label=la[4])
    plt.xlabel("Physical time")
    plt.title("{}".format(tag))
    if (log): plt.yscale("log")
    plt.legend()
    plt.savefig("ana_{}.png".format(tag))

def compare_M():
    col=6
    plt.figure()
    for k in data:
        y = np.sqrt(data[k][:,col]**2+data[k][:,col+1]**2+data[k][:,col+2]**2)
        plt.plot(data[k][:,0], y, label=k.decode())
    plt.xlabel("Physical time")
    plt.title("M")
    #if (log): plt.yscale("log")
    plt.legend()
    plt.savefig("ana_M.png".format(tag))

#:maxrelP,    2:surv, survb,    4:avgP, avgPb,      6:aM0
compare(1,"max(P-1)", 1)
compare(2,"Pee")
compare(3,"Peeb")
compare(4,"avgP")
compare(5,"avgPb")
compare_M()