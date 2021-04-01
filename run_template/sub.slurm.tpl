#!/bin/bash
#SBATCH --job-name ${JNAME}
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 192
#SBATCH --mem-bind=verbose,p

echo "SLURM_JOBID        : " $$SLURM_JOBID
echo "SLURM_JOB_NODELIST : " $$SLURM_JOB_NODELIST
echo "SLURM_NNODES       : " $$SLURM_NNODES
echo "working directory  : " $$SLURM_SUBMIT_DIR
echo "Start: " `date`
echo "========================================"

module purge
module load ThunderX2CN99/RHEL/7/gcc-9.3.0/armpl

srun --cpu-bind=v,cores ../nuosc --dz ${DZ} --zmax ${ZMAX} --cfl ${CFL} --nvz ${NVZ} --ko ${KO}
echo "--- Walltime: $${SECONDS} sec."
                
bash ./go_zmap
