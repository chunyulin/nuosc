
#!/bin/bash

sub_t3() {
    local fname=`basename $BASH_SOURCE`;
    local runtag=$1; shift

    local node=$1; shift
    local task=$1; shift
    local core=$1; shift

    local px=$1; shift
    local py=$1; shift
    local pz=$1; shift

    local xm=$1; shift
    local ym=$1; shift
    local zm=$1; shift
    local nv=$1; shift
    local nphi=$1; shift

    local bind=$1; shift
    local map=$1; shift

    printf -v n0 "%03d" $node
    printf -v t0 "%03d" $task
    printf -v c0 "%03d" $core
    local jobtag=n${n0}x${t0}c${c0}_s${d}v${nv}p${nphi}
    local folder=${runtag}/${jobtag}
    mkdir ${folder} -p

    cp ../nuosc ${fname} ${folder}

    local subfile=${jobtag}.slurm
    cat << EOF > ${folder}/${subfile}
#!/bin/bash
#SBATCH -A GOV109092
#SBATCH -J ${runtag}_${jobtag}
#SBATCH --nodes=${node} --ntasks-per-node=${task} --cpus-per-task=${core}
#SBATCH --mem-bind=verbose,p
#SBATCH --time=10:0

module purge
source /home/p00lcy01/t3pkg/nv/openmpi-4.1.4/env.sh
module load libs/GSL/2.6
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}

N=10
succ=0
for i in \$(seq 0 \$N); do
  echo
  echo "============== Try MPI \$i / \$N ================"
  mpirun --bind-to ${bind} --map-by ${map} --mca routed radix -report-bindings -np \${SLURM_NTASKS} \
  ./nuosc --np ${px} ${py} ${pz} --pmo 0 --mu 1 --ko 1e-3 --ipt 0   --xmax ${xm} ${ym} ${zm} --dx 0.1 \
          --nv ${nv} --nphi ${nphi}  --cfl 0.5 --alpha 0.9 --eps0 1e-3 --sigma 5.0   --ANA_EVERY 10 --END_STEP 10
  if [ "\$?" == "0" ]; then
     echo "---- Walltime: \${SECONDS} sec."
     ((succ=succ+1))
     if [ "\$succ" == "5" ]; then
        break
     fi
  fi
  echo "---- Walltime: \${SECONDS} sec."
done
EOF
    (cd ${folder}; sbatch ${subfile} )
}

test_weak() {
 local tag=$1; shift
 sub_t3 ${tag}  1  1 56   1 1 1   3  3  3 18 18 none node
 sub_t3 ${tag}  1  2 28   2 1 1   3  3  3 18 18 socket socket
 #sub_t3 ${tag}  2  1 56   2 1 1   6  3  3 18 18 none node
 sub_t3 ${tag}  2  2 28   2 2 1   6  3  3 18 18 socket socket
 #sub_t3 ${tag}  4  1 56   2 2 1   6  6  3 18 18 none node
 sub_t3 ${tag}  4  2 28   2 2 2   6  6  3 18 18 socket socket
}

test_strong() {
 local tag=$1; shift
 sub_t3 ${tag}  1  1 56   1 1 1   6 6 6 18 18 none node
 sub_t3 ${tag}  1  2 28   2 1 1   6 6 6 18 18 socket socket
 sub_t3 ${tag}  2  1 56   2 1 1   6 6 6 18 18 none node
 sub_t3 ${tag}  2  2 28   2 2 1   6 6 6 18 18 socket socket
 sub_t3 ${tag}  4  1 56   2 2 1   6 6 6 18 18 none node
 sub_t3 ${tag}  4  2 28   2 2 2   6 6 6 18 18 socket socket
 sub_t3 ${tag}  8  1 56   2 2 2   6 6 6 18 18 none node
 sub_t3 ${tag}  8  2 28   4 2 2   6 6 6 18 18 socket socket
}

test_strong strong_6_6_6_18

