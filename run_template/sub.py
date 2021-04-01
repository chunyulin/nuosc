from __future__ import print_function
import os
from string import Template
import subprocess

tpl = Template( open( 'sub.slurm.tpl', 'r').read() )

mu = 1.0

ZMAX = {512.0}
CFL = {0.25}
DZ = {0.25}
NVZ ={13}
KO ={1e-5, 1e-6}

###
### generate sub script and run in sub folders
###

for zmax in ZMAX:
 for cfl in CFL:
  for dz in DZ:
    for nvz in NVZ:
     for ko in KO:
        dirname = "ko{}_nvz{}".format(ko, nvz)
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

                
                