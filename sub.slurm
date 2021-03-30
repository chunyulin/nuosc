#!/bin/bash
#SBATCH --job-name NuOsc_512adv
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 224

module purge
module load ThunderX2CN99/RHEL/7/gcc-9.3.0/armpl


DZ="0.25 0.5 1"

for i in ${DZ}; do
    DIR=dz${i}
    mkdir -p ${DIR}
    (cd ${DIR}; srun --cpu-bind=v,cores ../nuosc --dz ${i}   --zmax 512.0 --cfl 0.25 )
    echo "--- Completed in ${SECONDS} sec."
done
