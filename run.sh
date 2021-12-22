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
    local ko=$1; shift
    local ana=$1; shift
    local dump=$1; shift
    local end=$1; shift

    local jobtag="r${alpha}k${ko}n${nvz}d${dz}"
    local folder="${runtag}/${jobtag}"
    mkdir ${folder} -p
    cp ./nuosc ./run.sh ${runtag}
    
    cat << EOF > ${folder}/sub.slurm
#!/bin/bash
#SBATCH --job-name ${jobtag}
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 240
#SBATCH --mem-bind=verbose,p
module purge
module load arm21/21.1
srun --cpu-bind=v,cores \
  ../nuosc --mu 1.0 --nv ${nvz} --dz ${dz} --zmax ${zmax} --cfl ${cfl} \\
  --ipt ${ipt} --eps0 ${eps0} --alpha ${alpha} --ko ${ko} \\
  --ANA_EVERY_T ${ana} --DUMP_EVERY_T ${dump} --ENDSTEP_T ${end}
echo "--- Walltime: \${SECONDS} sec."
EOF
    (cd ${folder}; sbatch sub.slurm )
}


ko () {
  group="ko$1"
  cfl=0.4
  zmax=6000
  eps0=1.e-6
  anatime=50.0
  dumptime=10000000.0
  endtime=6000.0
  ko=1.0
  nv=101
  dz=0.5
  alpha=1.0
  
  ipt=$1
sub ${group} ${cfl} ${zmax}  ${dz} ${nv} ${ipt} ${eps0} ${alpha} 1.0 ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  ${dz} ${nv} ${ipt} ${eps0} ${alpha} 0.1 ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  ${dz} ${nv} ${ipt} ${eps0} ${alpha} 0.01 ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  ${dz} ${nv} ${ipt} ${eps0} ${alpha} 0.001 ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  ${dz} ${nv} ${ipt} ${eps0} ${alpha} 0.0001 ${anatime} ${dumptime} ${endtime}
}


ko 0  ## point-like
#alpha 1  ## random
