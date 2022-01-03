#!/bin/bash
subarm() {
    local runset=$1; shift
    local jobid=$1; shift
    local exec=$*

    echo $exec
        
    local jobfolder="${runset}/${jobid}"
    mkdir ${jobfolder} -p
    cp ./nuosc ./run_twcc.sh ${runset}
    
    cat << EOF > ${jobfolder}/sub.slurm
#!/bin/bash
#SBATCH -A GOV109018 -p gp4d
#SBATCH --job-name ${jobid}
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 4
#SBATCH --gres=gpu:1
###SBATCH --mem-bind=verbose,p
module purge
module load nvhpc/21.7
srun --cpu-bind=v,cores ${exec}
echo "--- Walltime: \${SECONDS} sec."

module load miniconda3 ffmpeg
conda activate /opt/ohpc/pkg/kagra/ENV/py37
MP4=${jobid}.mp4
python3 ../../pltZVz_P3.py P3ZVz_*.bin
convert -delay 5 ZY*.png \${MP4}
chmod a+r \${MP4}
EOF
    (cd ${jobfolder}; sbatch sub.slurm )
}

comp2d () {
  RUNSET="comp2d_zmax$2"
  JOBID="y$1_nv$3_a$4"
  EXEC="../nuosc --ymax $1 --zmax $2 --dz 0.1 --nv $3 --cfl 0.4 --ko 0.1
             --sy $(($1*10)) --sz 2560 --sv $3
             --mu 1.0 --ipt 0 --eps0 1e-2 --alpha $1
             --ANA_EVERY_T 2 --DUMP_EVERY_T 2 --END_STEP_T $(($2*4))"

  subarm ${RUNSET} ${JOBID} ${EXEC}
}

ipt2 () {
  RUNSET="ipt2z$2"
  JOBID="ipt2_a$1_$nv$3_s$4"
  EXEC="../nuosc --ymax $2 --zmax $2 --dz 0.1 --nv $3 --cfl 0.4 --ko 0.1
             --sy $(($2*10)) --sz $(($2*10)) --sv $3
             --mu 1.0 --ipt 2 --eps0 1e-2 --alpha $1 --sigma $4 
             --ANA_EVERY_T 2 --DUMP_EVERY_T 2 --END_STEP_T $(($2*3))"

  subarm ${RUNSET} ${JOBID} ${EXEC}
}


#ipt2 0.9 32 20 4
#ipt2 0.9 50 16 4
#ipt2 1.1 50 16 4

comp2d 2 512 20 0.9
comp2d 2 512 20 1.0
comp2d 2 512 20 1.1
comp2d 2 512 20 1.2

comp2d 2 512 32 0.9
comp2d 2 512 32 1.0
comp2d 2 512 32 1.1
comp2d 2 512 32 1.2

comp2d 2 256 32 0.9
comp2d 2 256 32 1.0
comp2d 2 256 32 1.1
comp2d 2 256 32 1.2
