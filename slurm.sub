#! /bin/bash
#SBATCH -A GOV109092
#SBATCH -p gtest
#SBATCH -J nuosc
#SBATCH --nodes=1 --ntasks-per-node=4 --cpus-per-task=4
#SBATCH --gres=gpu:4

module purge
##export OMP_NUM_THREADS=16
module load nvhpc/21.7

NSYS=/work/opt/ohpc/pkg/qchem/nv/nsight-systems-2020.3.1/bin/nsys
OUT=nuosc
##.%q{PMIX_RANK}

srun ${NSYS} profile -o ${OUT} -f true --trace openmp,nvtx,cuda ./nuosc
#srun ./nuosc --dz 0.005 --nvz 151 --mu 0.1 --ANA_EVERY 999

echo "--- Walltime: ${SECONDS} sec."
