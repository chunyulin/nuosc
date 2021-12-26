#!/pkg/ENV/pycbc_py2/bin/python
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 9})

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys

P1D="p1_v.dat"
P2D="p2_v.dat"
P3D="p3_v.dat"


def read(folder):
    fn="./{}/{}".format(folder,P1D)
    with open(fn) as f:
	nz, z0, z1 = np.array( f.readline().split(' ')).astype(np.float)
    dz=(z1-z0)/nz
    #Z = np.linspace(z0+0.5*dz,z1-0.5*dz,nz)
    p1 = np.loadtxt("./{}/{}".format(folder,P1D), skiprows=1)
    p2 = np.loadtxt("./{}/{}".format(folder,P2D), skiprows=1)
    p3 = np.loadtxt("./{}/{}".format(folder,P3D), skiprows=1)
    p1[:,1:] = np.sqrt(p1[:,1:]*p1[:,1:] + p2[:,1:]*p2[:,1:])
    return nz,z0,z1, p1, p3


p = subprocess.check_output(['ls -d G*'],shell=True)
runs = p.split()

for run in runs:

    nz,z0,z1,p12,p3 = read(run)
    print("Plotting {} in z [{}:{}] ({})".format(run, z0, z1, nz))

    ## p12
    plt.figure(figsize=(8, 6))
    im = plt.imshow(p12[:,1:], extent=[float(z0), float(z1), p12[0,0], p12[-1,0]], origin='lower', aspect='auto', interpolation='none')
    plt.colorbar(im)
    plt.xlabel("Z")
    plt.ylabel("Phy time")
    plt.title(run)

    plt.plot([z0,0,z1],[np.abs(z0), 0, np.abs(z1)], 'r--')

    plt.savefig("P12zt_{}.png".format(run))
    plt.close()
    
    ## p3
    plt.figure(figsize=(8, 6))
    im = plt.imshow(p3[:,1:], extent=[float(z0), float(z1), p12[0,0], p12[-1,0]], origin='lower', aspect='auto', interpolation='none')
    plt.colorbar(im)
    plt.xlabel("Z")
    plt.ylabel("Phy time")
    plt.title(run)

    plt.plot([z0,0,z1],[np.abs(z0), 0, np.abs(z1)], 'r--')

    plt.savefig("P3zt_{}.png".format(run))
    plt.close()
    
