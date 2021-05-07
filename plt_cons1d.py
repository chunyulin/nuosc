#!/pkg/ENV/pycbc_py2/bin/python

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 10})

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import subprocess

p = subprocess.check_output(['ls -d G3a_r*'],shell=True)
runs = p.split()

data=dict()
for run in runs:
    data[run]=np.loadtxt("./{}/analysis.dat".format(run))
#d2=np.loadtxt("./G3a_0_0.4_0.1/analysis.dat")
#d3=np.loadtxt("./G3a_1_0.2_0.1/analysis.dat")
#d4=np.loadtxt("./G3a_1_0.4_0.1/analysis.dat")


def compare(col, tag):
    plt.figure()
    for k in data:
        plt.plot(data[k][:,0],(data[k][:,col]-data[k][0,col]), label=k)
    #plt.plot(d2[:,0],(d2[:,col]-d2[0,col]), label=la[2])
    #plt.plot(d3[:,0],(d3[:,col]-d3[0,col]), label=la[3])
    #plt.plot(d4[:,0],(d4[:,col]-d4[0,col]), label=la[4])
    plt.xlabel("Physical time")
    plt.title("G3a {}".format(tag))
    #plt.yscale("log")
    plt.legend()
    plt.savefig("compare_{}.png".format(tag))


compare(1,"avgP")
compare(2,"avgPb")
compare(3,"amaxrelP")
compare(4,"amaxrelN")
compare(5,"aM0")
compare(6,"aM1")
