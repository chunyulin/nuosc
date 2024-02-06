
#!/bin/bash

sub_t2() {
    local fname=`basename $BASH_SOURCE`;
    local runtag=$1; shift

    local node=$1; shift
    local gpu=$1; shift
    local task=${gpu}
    local core=4

    local px=$1; shift
    local py=$1; shift
    local pz=$1; shift

    local xm=$1; shift
    local ym=$1; shift
    local zm=$1; shift
    local nv=$1; shift
    local nphi=$1; shift

    printf -v n0 "%01d" $node
    local jobtag=n${n0}g${gpu}_x${xm}y${ym}z${zm}v${nv}p${nphi}
    local folder=${runtag}/${jobtag}
    mkdir ${folder} -p

    cp ./nuosc ${fname} ${folder}

    local subfile=${jobtag}.slurm
    cat << EOF > ${folder}/${subfile}
#!/bin/bash
#SBATCH -A ENT211349
#SBATCH -J ${runtag}_${jobtag}
#SBATCH --nodes=${node} --ntasks-per-node=${gpu} --cpus-per-task=${core}
#SBATCH --mem-bind=verbose,p
#SBATCH --gres=gpu:${gpu}
#SBATCH --time=10:0

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}

module purge
source /opt/ohpc/pkg/kagra/nv/openmpi-4.1.4_cuda/env.sh

##--mca routed radix
mpirun --bind-to none --report-bindings -np \${SLURM_NTASKS} \
 ./nuosc --np ${px} ${py} ${pz} --pmo 0 --mu 1 --ko 1e-3 --ipt 0 \
    --xmax ${xm} ${ym} ${zm} --dx 0.1 --nv ${nv} --nphi ${nphi} --cfl 0.5 --alpha 0.9 --eps0 1e-3 --sigma 5.0 \
    --ANA_EVERY 5 --END_STEP 10

if [ "$?" == "0" ]; then
    echo "Completed."
else
    echo "Something wrong."
fi
echo "--- Walltime: \${SECONDS} sec."
EOF
    (cd ${folder}; sbatch ${subfile} )
}

test_weak() {
 local tag=$1; shift
 sub_t2 ${tag} 1 1    1 1 1      3 3 3   18 18
 sub_t2 ${tag} 1 2    2 1 1      6 3 3   18 18
 sub_t2 ${tag} 1 3    3 1 1      9 3 3   18 18
 sub_t2 ${tag} 1 4    2 2 1      6 6 3   18 18
 sub_t2 ${tag} 1 5    5 1 1     15 3 3   18 18
 sub_t2 ${tag} 1 6    3 2 1      9 6 3   18 18
 sub_t2 ${tag} 1 7    7 1 1     21 3 3   18 18
 sub_t2 ${tag} 1 8    2 2 2      6 6 6   18 18
}

test_weak2n() {
 local tag=$1; shift
 sub_t2 ${tag} 2 1    2 1 1      6 3 3   18 18
 sub_t2 ${tag} 2 2    2 2 1      6 6 3   18 18
 sub_t2 ${tag} 2 3    3 2 1      9 6 3   18 18
 sub_t2 ${tag} 2 4    2 2 2      6 6 6   18 18
 sub_t2 ${tag} 2 5    5 2 1     15 6 3   18 18
 sub_t2 ${tag} 2 6    3 2 2      9 6 6   18 18
 sub_t2 ${tag} 2 7    7 2 1     21 6 3   18 18
 sub_t2 ${tag} 2 8    4 2 2     12 6 6   18 18
}
test_weak3n() {
 local tag=$1; shift
 sub_t2 ${tag} 3 1    3 1 1      9 3 3   18 18
 sub_t2 ${tag} 3 2    3 2 1      9 6 3   18 18
 sub_t2 ${tag} 3 3    3 3 1      9 9 3   18 18
 sub_t2 ${tag} 3 4    3 2 2      9 6 6   18 18
 sub_t2 ${tag} 3 5    5 3 1     15 9 3   18 18
 sub_t2 ${tag} 3 6    3 3 2      9 9 6   18 18
 sub_t2 ${tag} 3 7    7 3 1     21 9 3   18 18
 sub_t2 ${tag} 3 8    6 2 2     18 6 6   18 18
}
test_weak4n() {
 local tag=$1; shift
 sub_t2 ${tag} 4 1    2 2 1      6  6 3   18 18
 sub_t2 ${tag} 4 2    2 2 2      6  6 6   18 18
 sub_t2 ${tag} 4 3    3 2 2      9  6 6   18 18
 sub_t2 ${tag} 4 4    4 2 2     12  6 6   18 18
 sub_t2 ${tag} 4 5    5 2 2     15  6 6   18 18
 sub_t2 ${tag} 4 6    4 3 2     12  9 6   18 18
 sub_t2 ${tag} 4 7    7 2 2     21  6 6   18 18
 sub_t2 ${tag} 4 8    4 4 2     12 12 6   18 18
}

test_weak5n() {
 local tag=$1; shift
 sub_t2 ${tag} 5 1    5 1 1     15  3  3  18 18
 sub_t2 ${tag} 5 2    5 2 1     15  6 3  18 18
 sub_t2 ${tag} 5 3    5 3 1     15  9 3  18 18
 sub_t2 ${tag} 5 4    5 2 2     15  6 6  18 18
 sub_t2 ${tag} 5 5    5 5 1     15  15 3  18 18
 sub_t2 ${tag} 5 6    5 3 2     15  9  6  18 18
 sub_t2 ${tag} 5 7    7 5 1     21  15 3  18 18
 #sub_t2 ${tag} 5 8    5 4 2     15  12 6  18 18
 #sub_t2 ${tag} 5 8    10 2 2    30  6  6   18 18
}

test_large() {
 local tag=$1; shift
 sub_t2 ${tag} 1 1    1 1 1     3 3 3   20 20
 sub_t2 ${tag} 1 1    1 1 1     5 3 3   18 18
 sub_t2 ${tag} 1 1    1 1 1     3 3 3   22 22
}
test_weak4n weak4n_gpu

