from __future__ import print_function
import os
from string import Template
import subprocess

tpl = Template( open( 'sub.slurm.tpl', 'r').read() )

CFL = [0.25]
NVZ = [13]
ZMAX = [512.0]
DZ = [0.25]
MU = [0.0]
KO =[0.0, 0.01, 0.001, 1e-3, 1e-4]

###
### generate sub script and run in sub folders
###

nvz = NVZ[0]

for cfl in CFL:
 for zmax in ZMAX:
  for dz in DZ:
     for ko in KO:
      for mu in MU:
        dirname = "ko5_{}_mu{}".format(ko, mu)
	subname = "sub.slurm"
	fullsub = "{}/{}".format(dirname, subname)
	
	try:
    	    os.mkdir(dirname)
	except:
	    pass

        fo = open(fullsub, "w")
	fo.write( tpl.substitute(DZ=dz, NVZ=nvz, CFL=cfl, ZMAX=zmax, KO=ko, JNAME=dirname, MU=mu) )
	fo.close()
	
	#os.system("sbatch "+ subname )
        p = subprocess.Popen(["cp", "../go_zmap", "./"], cwd=dirname)
        p = subprocess.Popen(["sbatch", subname], cwd=dirname)

                
                