#!/bin/bash
sub_t4() {
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

    ##local bind=$1; shift

    printf -v n0 "%03d" $node
    printf -v t0 "%03d" $task
    printf -v c0 "%03d" $core
    local jobtag=n${n0}x${t0}c${c0}_v${nv}p${nphi}
    local folder=${runtag}/${jobtag}
    mkdir ${folder} -p

    cp ../nuosc ${fname} ${folder}

    local subfile=${jobtag}.slurm
    cat << EOF > ${folder}/${subfile}
#!/bin/bash
#SBATCH -A GOV113006 -p alphatest
#SBATCH -J ${runtag}_${jobtag} -t 100:0
#SBATCH --nodes=${node} --ntasks-per-node=${task} --cpus-per-task=${core}
#SBATCH --mem-bind=no
##export OMP_PROC_BIND=true
##export OMP_PLACES=cores
##export OMP_DYNAMIC=false
VARS="SLURM_JOB_ID SLURM_NTASKS SLURM_NNODES SLURM_NTASKS_PER_NODE SLURM_CPUS_PER_TASK SLURM_TASK_PID SLURM_NODELIST \
SLURM_SUBMIT_DIR  SLURM_MEM_BIND SLURM_MPI_TYPE \
OMP_PROC_BIND OMP_PLACES OMP_DYNAMIC"
for v in \${VARS}; do echo " \$v = \${!v}"; done

ml purge
#ml openmpi/5.0.1
ml sys/intelmpi/2021.9

N=5
for i in \$(seq 1 \$N); do
  echo "============== Try MPI \$i / \$N ================"
  mpiexec \
  ./nuosc --np ${px} ${py} ${pz} --pmo 0 --mu 1 --ko 1e-3 --ipt 0   --xmax ${xm} ${ym} ${zm} --dx 0.1 \
          --nv ${nv} --nphi ${nphi}  --cfl 0.5 --alpha 0.9 --eps0 1e-3 --sigma 5.0   --ANA_EVERY 10 --END_STEP 10
  echo "---- Walltime: \${SECONDS} sec."
done
EOF
    (cd ${folder}; sbatch ${subfile} )
}

test_weak_power2() {
 local tag=$1; shift
 local b=$1; shift
 local nv=$1; shift
 local nphi=$1; shift
 sub_t4 ${tag}  1    1 112   1 1 1  $((b*1)) $((b*1)) $((b*1))  $nv $nphi
 sub_t4 ${tag}  2    1 112   2 1 1  $((b*2)) $((b*1)) $((b*1))  $nv $nphi
 sub_t4 ${tag}  4    1 112   2 2 1  $((b*2)) $((b*2)) $((b*1))  $nv $nphi
 sub_t4 ${tag}  8    1 112   2 2 2  $((b*2)) $((b*2)) $((b*2))  $nv $nphi
 sub_t4 ${tag}  16   1 112   4 2 2  $((b*4)) $((b*2)) $((b*2))  $nv $nphi
 sub_t4 ${tag}  32   1 112   4 4 2  $((b*4)) $((b*4)) $((b*2))  $nv $nphi
 sub_t4 ${tag}  64   1 112   4 4 4  $((b*4)) $((b*4)) $((b*4))  $nv $nphi
 sub_t4 ${tag}  128  1 112   8 4 4  $((b*8)) $((b*4)) $((b*4))  $nv $nphi
 sub_t4 ${tag}  256  1 112   8 8 4  $((b*8)) $((b*8)) $((b*4))  $nv $nphi
 sub_t4 ${tag}  512  1 112   8 8 8  $((b*8)) $((b*8)) $((b*8))  $nv $nphi
}

test_weak_1d() {
 local tag=$1; shift
 local b=$1; shift
 local nv=$1; shift
 local nphi=$1; shift
 sub_t4 ${tag}    2  1 112    1 1 2  $((b*1)) $((b*1)) $((b*1))  $nv $nphi
 sub_t4 ${tag}    4  1 112    1 2 2  $((b*1)) $((b*1)) $((b*1))  $nv $nphi
 sub_t4 ${tag}    8  1 112    1 4 2  $((b*1)) $((b*1)) $((b*1))  $nv $nphi
 sub_t4 ${tag}   15  1 112    1 5 3  $((b*1)) $((b*1)) $((b*1))  $nv $nphi
 sub_t4 ${tag}   30  1 112    2 5 3  $((b*2)) $((b*1)) $((b*1))  $nv $nphi
 sub_t4 ${tag}   60  1 112    4 5 3  $((b*2)) $((b*2)) $((b*1))  $nv $nphi
 sub_t4 ${tag}  120  1 112    8 5 3  $((b*2)) $((b*2)) $((b*2))  $nv $nphi
 sub_t4 ${tag}  240  1 112   16 5 3  $((b*4)) $((b*2)) $((b*2))  $nv $nphi
 sub_t4 ${tag}  360  1 112   24 5 3  $((b*4)) $((b*4)) $((b*2))  $nv $nphi
 sub_t4 ${tag}  450  1 112   30 5 3  $((b*4)) $((b*4)) $((b*2))  $nv $nphi
 #sub_t4 ${tag}  540  1 112   36 5 3  $((b*8)) $((b*4)) $((b*4))  $nv $nphi
}

test_strong() {
 local tag=$1; shift
 local b=$1; shift
 local nv=$1; shift
 local nphi=$1; shift
 sub_t4 ${tag}   1  1 112   1 1 1   $b $b $b $nv $nphi
 sub_t4 ${tag}   2  1 112   2 1 1   $b $b $b $nv $nphi
 sub_t4 ${tag}   4  1 112   2 2 1   $b $b $b $nv $nphi
 sub_t4 ${tag}   8  1 112   2 2 2   $b $b $b $nv $nphi
 sub_t4 ${tag}  16  1 112   4 2 2   $b $b $b $nv $nphi
 sub_t4 ${tag}  32  1 112   4 4 2   $b $b $b $nv $nphi
 sub_t4 ${tag}  64  1 112   4 4 4   $b $b $b $nv $nphi
 sub_t4 ${tag} 128  1 112   8 4 4   $b $b $b $nv $nphi
 sub_t4 ${tag} 256  1 112   8 8 4   $b $b $b $nv $nphi
 sub_t4 ${tag} 512  1 112   8 8 8   $b $b $b $nv $nphi
}

test_one() {
 local tag=$1; shift
 local px=$1; shift
 local py=$1; shift
 local pz=$1; shift
 local b=$1; shift
 local nv=$1; shift
 local nphi=$1; shift
 sub_t4 ${tag}  $((px*py*pz))    1 112   $px $py $pz  $((b*px)) $((b*py)) $((b*pz))  $nv $nphi
}

#test_weak_1d   w1d4 4 18 18
#test_weak_1d   w1d6 6 18 18
test_weak_1d   w1d7 7 18 18

#test_one one7 1 1 1   7   18 18

#test_weak   pweak4  20 20
#test_weak   pweak5 5 20 20
#test_weak   pweak6 6 20 20
#test_weak   pweak7 7 18 18
#test_strong pstro4 4 18 18
#test_strong pstro5 5 18 18
#test_strong pstro6 6 18 18
#test_strong pstro7 7 18 18

