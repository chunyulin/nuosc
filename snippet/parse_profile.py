#!/usr/bin/env python3
import glob
import numpy as np
import os, sys, subprocess


def profile(fname):

 breakdowns = dict()
 breakdowns['calRHS'] = []

 t=0
 f=open(fname)
 line = f.readline()
 
 while line:
  if t==1:  ## calRHS
    if line.startswith("  END"):
       x=line.split()
       if x[1] not in breakdowns:
          breakdowns[x[1]]=[float(x[4])]
       else:
          breakdowns[x[1]].append( float(x[4]) )
  if line.startswith(" BEG calRHS "):
   t=1
  elif line.startswith(" END calRHS "):
   x=line.split()
   breakdowns['calRHS'].append( float(x[4]) )
  line = f.readline()

 #for k in breakdowns:
 #   arr = np.array( breakdowns[k] )
 #   print("{:<20}: {:8.1f} ± {:7.1f} in [ {:7.1f} {:7.1f} ]".format(k,arr.mean(), arr.std(), arr.min(), arr.max()) )

 for k in breakdowns:  breakdowns[k] = np.array( breakdowns[k] )
 t0 = breakdowns['calRHS'].mean()
 t1 = breakdowns['calRHS_with_bdry'].mean()
 t2 = breakdowns['calRHS_wo_bdry'].mean()
 t3 = (breakdowns['Packing'] + breakdowns['Unpacking']).mean()
 t4 = (breakdowns['Sync']+ breakdowns['Waitall']).mean()
 print("{:<15} {:4.1f} {:4.1f} {:4.1f} {:4.1f}".format(fname, t1/t0*100, t2/t0*100, t3/t0*100, t4/t0*100))
 #print(breakdowns)

if __name__ == '__main__':
  fname = glob.glob("profile.*")
  for f in fname:
    profile(f)



