#!/pkg/ENV/pycbc_py2/bin/python

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 9})

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys

def read(folder):
    fn="./{}/p1_vm.dat".format(folder)
    with open(fn) as f:
	nz, z0, z1 = np.array( f.readline().split(' ')).astype(np.float)
    
    dz=(z1-z0)/nz
    Z = np.linspace(z0+0.5*dz,z1-0.5*dz,nz)
    p1 = np.loadtxt("./{}/p1_v.dat".format(folder), skiprows=1)
    p2 = np.loadtxt("./{}/p2_v.dat".format(folder), skiprows=1)
    p3 = np.loadtxt("./{}/p3_v.dat".format(folder), skiprows=1)
    #p1[:,1:] = np.sqrt(p1[:,1:]*p1[:,1:] + p2[:,1:]*p2[:,1:])
    return Z, p1, p2, p3


p = subprocess.check_output(['ls -d G*'],shell=True)
runs = p.split()

p1=dict()
p2=dict()
p3=dict()
Z=dict()
for run in runs:
    Z[run],p1[run],p2[run],p3[run] = read(run)

## assume all have the same # nt dump
nt  = np.shape(p1[runs[0]])[0]
ntp = range(0,nt,nt//4)

for run in runs:
  print(run)

  plt.figure(figsize=(10, 5))
  
  offset = 0
  for i in ntp:
    plt.plot(Z[run], p3[run][i,1:] + offset, label=p3[run][i,0])
    offset += 2

  plt.title("P3")
  plt.xlabel("Z")
  #plt.yscale("log")
  #plt.ylim([1e-9,0.01])
  plt.legend()
  #plt.grid()

  plt.savefig("Pprof_{}.png".format(run))
  plt.close()

