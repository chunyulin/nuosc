#!/bin/bash
#SBATCH --job-name nuosc_test
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 192

module purge
module load ThunderX2CN99/RHEL/7/gcc-9.3.0/armpl


DZ="0.5"
NVZ="9 15 51 71"

for dz in ${DZ}; do
for nvz in ${NVZ}; do
    DIR=dz${dz}_nvz${nvz}
    mkdir -p ${DIR}
    (cd ${DIR}; srun --cpu-bind=v,cores ../nuosc --dz ${dz}   --zmax 512.0 --cfl 0.25  --nvz ${nvz} )
    echo "--- Completed in ${SECONDS} sec."
done
done
