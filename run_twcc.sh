#!/bin/bash

sub() {
    local runtag=$1; shift
    local cfl=$1; shift
    local zmax=$1; shift
    local dz=$1; shift
    local nvz=$1; shift
    local ipt=$1; shift
    local eps0=$1; shift
    local alpha=$1; shift
    local sigma=$1; shift
    local ko=$1; shift
    local ana=$1; shift
    local dump=$1; shift
    local end=$1; shift

    local jobtag="r${alpha}_k${ko}_n${nvz}_d${dz}_eps${eps0}_s${sigma}"
    local folder="${runtag}/${jobtag}"
    mkdir ${folder} -p
    cp ./nuosc ./run_twcc.sh ${runtag}
    
    cat << EOF > ${folder}/sub.slurm
#!/bin/bash
#SBATCH --job-name ${runtag}_${renorm}_${dz}_${eps0}
#SBATCH -A GOV109092
#SBATCH -p gp1d
#SBATCH --gres=gpu:2
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 8
#SBATCH --mem-bind=verbose,p
module purge
module load nvhpc/21.7
srun --cpu-bind=v,cores \
  ../nuosc --mu 1.0 --nv ${nvz} --dz ${dz} --zmax ${zmax} --cfl ${cfl} \\
           --ipt ${ipt} --eps0 ${eps0} --alpha ${alpha} --sigma ${sigma} --ko ${ko} \\
           --ANA_EVERY_T ${ana} --DUMP_EVERY_T ${dump} --ENDSTEP_T ${end}
echo "--- Walltime: \${SECONDS} sec."
EOF
    (cd ${folder}; sbatch sub.slurm )
}


noc () {
  group="noc$1"
  cfl=0.4
  zmax=5120
  eps0=1.e-2
  anatime=5.0
  dumptime=10000000.0
  endtime=6000.0
  nv=201
  dz=0.1
  alpha=1.0
  sigma=100.0
  ipt=$1

sub ${group} ${cfl} ${zmax}  ${dz} ${nv} ${ipt} ${eps0} ${alpha} ${sigma} 1.0 ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  ${dz} ${nv} ${ipt} ${eps0} ${alpha} ${sigma} 0.5 ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  ${dz} 401   ${ipt} ${eps0} ${alpha} ${sigma} 1.0 ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.05  ${nv} ${ipt} ${eps0} ${alpha} ${sigma} 1.0 ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.2   ${nv} ${ipt} ${eps0} ${alpha} ${sigma} 1.0 ${anatime} ${dumptime} ${endtime}
}

noc 0  ## point-like
#alpha 1  ## random
