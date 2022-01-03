#!/bin/bash
subarm() {
    local runset=$1; shift
    local jobid=$1; shift
    local exec=$*

    echo $exec
        
    local jobfolder="${runset}/${jobid}"
    mkdir ${jobfolder} -p
    cp ./nuosc ./run.sh ${runset}
    
    cat << EOF > ${jobfolder}/sub.slurm
#!/bin/bash
#SBATCH --job-name ${jobid}
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 240
#SBATCH --mem-bind=verbose,p
module purge
module load arm21/21.1
srun --cpu-bind=v,cores ${exec}
echo "--- Walltime: \${SECONDS} sec."
EOF
    (cd ${jobfolder}; sbatch sub.slurm )
}


test1d () {
  RUNSET="folder"
  JOBID="z${zmax}"

  zmax=128
  EXEC="../nuosc --dz 0.1 --zmax ${zmax} --nv 200 --cfl 0.4 --ko 0.1
                 --mu 1.0 --ipt 0 --eps0 1e-2 --alpha 0.9
                 --ANA_EVERY_T 2 --DUMP_EVERY_T 2 --END_STEP_T $((2*zmax))"
  
  subarm ${RUNSET} ${JOBID} ${EXEC}

}


test1d