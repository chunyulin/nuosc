#!/bin/bash

run() {
    local renorm=$1
    local eps0=$2
    local alpha=$3
    local dz=$4
    local tag=$5
    
    id=${tag}_${renorm}_${dz}_${eps0}
    mkdir ${id}
    cp ./nuosc ${id}
    (cd ${id}; \
     numactl -C 0-127 ./nuosc \
     --mu 1.0 --nvz 51 --dz ${dz} --zmax 600 --cfl 0.4 \
     --renorm ${renorm} --eps0 ${eps0} --alpha ${alpha} \
     --ANA_EVERY_T 8.0 --ENDSTEP_T 1200.0 \
    )
}

sub() {
    local renorm=$1
    local eps0=$2
    local alpha=$3
    local dz=$4
    local runtag=$5
    
    folder="${runtag}/${runtag}_${renorm}_${dz}_${eps0}"
    mkdir ${folder} -p
    cp ./nuosc ${runtag}
    
    cat << EOF > ${folder}/sub.slurm
#!/bin/bash
#SBATCH --job-name ${runtag}_${renorm}_${dz}_${eps0}
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 60
#SBATCH --mem-bind=verbose,p
module purge
module load ThunderX2CN99/RHEL/7/gcc-9.3.0/armpl
srun --cpu-bind=v,cores ../nuosc --mu 1.0 --nvz 51 --dz ${dz} --zmax 600 --cfl 0.4 \
     --renorm ${renorm} --eps0 ${eps0} --alpha ${alpha} \
     --ANA_EVERY_T 8.0 --ENDSTEP_T 1200.0
echo "--- Walltime: \${SECONDS} sec."
EOF
    (cd ${folder}; sbatch sub.slurm )
}

sub 0  0.1  0.9  0.4 "G3a"
sub 0  0.1  0.9  0.2 "G3a"
sub 1  0.1  0.9  0.4 "G3a"
sub 1  0.1  0.9  0.2 "G3a"

