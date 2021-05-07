import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def plot_pp(tag, fname):
    NLINE = 5
    basename = os.path.basename(fname).split('.')[0]
    with open(fname) as f:
	nz, z0, z1 = np.array( f.readline().split(' ')).astype(np.float)
    dz=(z1-z0)/nz
    Z = np.linspace(z0+0.5*dz,z1-0.5*dz,nz)

    data = np.loadtxt(fname, skiprows=1)
    nt    = np.shape(data)[0]
    pltnt = range(1,nt, nt//NLINE)
    
    ## 1d
    plt.figure(figsize=(8, 6))
    for i in [0]+pltnt:
	print(i)
        plt.plot(Z, data[i,1:], label=data[i,0], linewidth=0.9)
    plt.xlabel("Z")
    plt.ylabel(tag)
    lg=plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("{}.jpg".format(tag), bbox_inches='tight', bbox_extra_artists=(lg,))
    plt.close()

def plot_P_perp(tag, p1f, p2f, p3f):
    NLINE = 5
    
    with open(p1f) as f:
	nz, z0, z1 = np.array( f.readline().split(' ')).astype(np.float)
    dz=(z1-z0)/nz
    Z = np.linspace(z0+0.5*dz,z1-0.5*dz,nz)

    p1 = np.loadtxt(p1f, skiprows=1)
    p2 = np.loadtxt(p2f, skiprows=1)
    p3 = np.loadtxt(p3f, skiprows=1)
    
    nt    = np.shape(p1)[0]
    pltnt = range(nt//NLINE, nt, nt//NLINE)
    
    ## 1d
    plt.figure(figsize=(8, 6))
    for i in [0]+pltnt:
	print(i)
        plt.plot(Z, np.sqrt(p1[i,1:]*p1[i,1:]+p2[i,1:]*p2[i,1:]), label=p1[i,0], linewidth=0.9)
        #plt.plot(Z, p3[i,1:], label=p3[i,0], linewidth=0.9)
    
    plt.xlabel("Pperp")
    plt.xlabel("Z")
    plt.yscale("log")
    plt.ylabel(tag)
    plt.ylim([1e-9,0.01])
    lg=plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("{}.jpg".format(tag), bbox_inches='tight', bbox_extra_artists=(lg,))
    plt.close()

def plot_P_conserved_1d(tag, p1f, p2f, p3f):
    
    with open(p1f) as f:
	nz, z0, z1 = np.array( f.readline().split(' ')).astype(np.float)
    dz=(z1-z0)/nz
    Z = np.linspace(z0+0.5*dz,z1-0.5*dz,nz)

    p1d = np.loadtxt(p1f, skiprows=1)
    p2d = np.loadtxt(p2f, skiprows=1)
    p3d = np.loadtxt(p3f, skiprows=1)
    
    t = p1d[:,0]
    p1 = p1d[:,1:]
    p2 = p2d[:,1:]
    p3 = p3d[:,1:]
    
    nt    = np.shape(p1)[0]
    
    ## 1d
    plt.figure(figsize=(8, 6))
    plt.plot(t, np.max(np.sqrt(p1*p1+p2*p2+p3*p3), axis=1), linewidth=0.9)
    
    plt.xlabel("T")
    plt.ylabel(tag)
    #plt.yscale("log")
    #plt.ylim([1e-9,0.01])
    lg=plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("{}.jpg".format(tag), bbox_inches='tight', bbox_extra_artists=(lg,))
    plt.close()

def plot1ds(tag, fname):
    NLINE = 4
    basename = os.path.basename(fname).split('.')[0]
    with open(fname) as f:
	nz, z0, z1 = np.array( f.readline().split(' ')).astype(np.float)
    dz=(z1-z0)/nz
    Z = np.linspace(z0+0.5*dz,z1-0.5*dz,nz)
    DS=10

    data = np.loadtxt(fname, skiprows=1)
    nt    = np.shape(data)[0]
    pltnt = range(1,nt, nt//NLINE)

    ## 1d
    plt.figure(figsize=(8, 6))
    for i in [0]+pltnt:
	print(i)
	plt.plot(Z[::DS], abs(data[i,1::DS]), label=data[i,0], linewidth=0.9)
    plt.xlabel("Z")
    plt.ylabel(tag)
    plt.yscale("log")
    lg=plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("{}.jpg".format(tag), bbox_inches='tight', bbox_extra_artists=(lg,))
    plt.close()

#tag, dat = sys.argv[1], sys.argv[2]
#plot1ds(tag, dat)

p1f = "../p1_v.dat"
p2f = "../p2_v.dat"
p3f = "../p3_v.dat"

plot_P_perp("Pperp", p1f, p2f, p3f)
plot_P_conserved_1d("P1d", p1f, p2f, p3f)
