from __future__ import print_function
import os
from string import Template
import subprocess

tpl = Template( open( 'sub.slurm.tpl', 'r').read() )

CFL = {0.25}
NVZ ={13}
ZMAX = {512.0}
DZ = {0.25}
KO ={1e-5, 1e-6}
MU = {0.0, 0.2, 0.5, 1.0}

###
### generate sub script and run in sub folders
###

cfl = CFL[0]
nvz = NVZ[0]

for zmax in ZMAX:
  for dz in DZ:
     for ko in KO:
      for mu in MU:
        dirname = "ko3_{}_mu{}".format(ko, mu)
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

                
                