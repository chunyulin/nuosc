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
    p1 = np.loadtxt("./{}/p1_vm.dat".format(folder), skiprows=1)
    p2 = np.loadtxt("./{}/p2_vm.dat".format(folder), skiprows=1)
    p1[:,1:] = np.sqrt(p1[:,1:]*p1[:,1:] + p2[:,1:]*p2[:,1:])
    return Z, p1


p = subprocess.check_output(['ls -d G*'],shell=True)
runs = p.split()
d=dict()
Z=dict()
for run in runs:
    Z[run],d[run] = read(run)

## assume all have the same # nt dump
nt = range(np.shape(d[runs[0]])[0])

for i in nt:
  print(d[runs[0]][i,0])

  plt.figure(figsize=(8, 6))
  for run in runs:
    plt.plot(Z[run], d[run][i,1:], label=run, linewidth=0.9)

  plt.title("Pperp")
  plt.xlabel("Z")
  plt.yscale("log")
  plt.ylim([1e-9,0.01])
  plt.yscale("log")
  plt.legend()
  plt.grid()

  plt.savefig("Pperp_{}.png".format(d[runs[0]][i,0]))
  plt.close()

