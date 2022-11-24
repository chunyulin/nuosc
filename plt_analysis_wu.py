#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 10})

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import subprocess

p = subprocess.check_output(['ls -d im_*_ex'],shell=True)
runs = p.split()

data=dict()
for run in runs:
    data[run]=np.loadtxt("./{}/analysis.dat".format(run.decode()), skiprows=1)
    #except:
    #    continue

wu = np.loadtxt("/home/lincy/codecomparison/finalG3a0.9_10240_200_0.4_4_1e-6_1_1e-1/m1.dat")

###
plt.xlabel("Physical time")

###
col=1
tag="max(P-1)"
plt.figure()
for k in data:
    plt.plot(data[k][:,0],data[k][:,col], label=k.decode())
plt.plot(wu[:,0],wu[:,3], label="Wu")
plt.title("{}".format(tag))
plt.yscale("log")
plt.legend()
plt.savefig("ana_{}.png".format(tag))

###
col=2
tag="Pee"
plt.figure()
for k in data:
    plt.plot(data[k][:,0],data[k][:,col], label=k.decode())
plt.plot(wu[:,0],wu[:,1], label="Wu")
plt.title("{}".format(tag))
#plt.yscale("log")
plt.legend()
plt.savefig("ana_{}.png".format(tag))

###
col=3
tag="Peeb"
plt.figure()
for k in data:
    plt.plot(data[k][:,0],data[k][:,col], label=k.decode())
plt.plot(wu[:,0],wu[:,2], label="Wu")
plt.title("{}".format(tag))

plt.legend()
plt.savefig("ana_{}.png".format(tag))

###
col=4
tag="AvgP"
plt.figure()
for k in data:
    plt.plot(data[k][:,0],data[k][:,col], label=k.decode())
plt.plot(wu[:,0],wu[:,7], label="Wu")
plt.title("{}".format(tag))

plt.legend()
plt.savefig("ana_{}.png".format(tag))

###
col=6
tag="|M|"
plt.figure()
for k in data:
    plt.plot(data[k][:,0],data[k][:,col]/10240, label=k.decode())
plt.plot(wu[:,0],wu[:,5], label="Wu")
plt.title("{}".format(tag))
plt.legend()
plt.savefig("ana_{}.png".format(tag))

###
col=8
tag="ELN_error"
plt.figure()
for k in data:
    plt.plot(data[k][:,0],data[k][:,col], label=k.decode())
plt.title("{}".format(tag))
plt.yscale("log")
plt.legend()
plt.savefig("ana_{}.png".format(tag))

###
col=9
tag="ELN_error2"
plt.figure()
for k in data:
    plt.plot(data[k][:,0],data[k][:,col], label=k.decode())
plt.plot(wu[:,0], abs( (1-wu[:,1] - 0.9*(1-wu[:,2]))/(1+0.9))     , label="Wu")
plt.title("{}".format(tag))
plt.yscale("log")
plt.legend()
plt.savefig("ana_{}.png".format(tag))

###
tag="ELN_error3"
for k in data:
    plt.figure()
    plt.plot(data[k][:,0],data[k][:,8], label="8")
    plt.plot(data[k][:,0],data[k][:,9], label="9")
    plt.title("{}".format(tag))
    plt.yscale("log")
    plt.legend()
    plt.savefig("ana_{}.png".format(k.decode()))


#:maxrelP,    2:surv, survb,    4:avgP, avgPb,      6:aM0
#compare(1,"max(P-1)", 1)
#compare(2,"Pee")
#compare(3,"Peeb")
#compare(4,"avgP")
#compare(5,"avgPb")
#compare(6,"|M|", 0)
#compare(7,"e-x")
#compare(8,"ELN_error", 1)
