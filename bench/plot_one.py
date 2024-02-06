#!/usr/bin/env python3
import matplotlib
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
import numpy as np
import os, sys, subprocess

BASE="T4_weak"
print(BASE)

#n016x001c112_sv18p18/slurm-745.out:[Summ] 112 4 2 2 100 100 100 324 45.422576 14.934254
def getdata(fname):
 d = np.loadtxt(fname, delimiter=' ', usecols=range(1,11))
 return np.array(d)

def genstat(d):
 """
 Output "[[ core  px py pz  nx ny nz nv  tp std min max]]"
 """
 stat = []
 for nq in np.unique(d[:,0:8],axis=0):
  mask_ = (d[0:,0:8]==nq)
  mask  = np.prod(mask_, axis=1, dtype=bool)
  tp = d[mask,8]
  stat += [ [i for i in nq] + [np.mean(tp), np.std(tp), np.min(tp), np.max(tp)] ]
 return np.array(stat)

d_ = getdata(BASE+".dat") #print(d_)


#==========
d  = genstat(d_)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
avg = d[:,8]
#plt.plot(d[:,1]*d[:,2]*d[:,3], d[:,8], 'r-', marker='.', ms=5, label="1 rank per node")
plt.errorbar(d[:,1]*d[:,2]*d[:,3], d[:,8], yerr=[ avg-d[:,10].T, d[:,11].T-avg ], capsize=5, marker=8, ms = 5, label="1 rank per node")
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend()
plt.xlabel("Nodes")
plt.ylabel("Per step grid-point time (ns)")
plt.title("{} w/ sys IMPI".format(BASE))
ax.set_xscale("log", base=2, nonpositive='clip')
#ax.set_yscale("log", nonpositive='clip')
ax.xaxis.set_major_formatter(ScalarFormatter())
#ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: ('{{}f}').format(x)))
#ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
plt.savefig(BASE+".png", bbox_inches='tight')

