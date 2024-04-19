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

    #local jobtag="r${alpha}k${ko}n${nvz}d${dz}"
    local jobtag="ipt${ipt}a${alpha}"
    local folder="${runtag}/${jobtag}"
    #local folder="${runtag}"
    mkdir ${folder} -p
    cp ./nuosc ./run.sh ${runtag}
    
    cat << EOF > ${folder}/sub.slurm
#!/bin/bash
#SBATCH --job-name ${jobtag}
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 240
#SBATCH --mem-bind=verbose,p
module purge
module load python/3.8.5
module load acfl/22.0.1
srun --cpu-bind=v,cores \
  ../nuosc --mu 1.0 --nv ${nvz} --dz ${dz} --zmax ${zmax} --cfl ${cfl} \\
  --ipt ${ipt} --eps0 ${eps0} --sigma 5.0 --alpha ${alpha} --ko ${ko} --lnue 0.6 0.5 \\
  --ANA_EVERY_T ${ana} --DUMP_EVERY_T ${dump} --END_STEP_T ${end}

python3 ~/nuosc_ben/plt_skimshot2d.py P*.bin
convert -delay 30 P*.png ${jobtag}.mpg
chmod a+r ${jobtag}.mpg
cp ${jobtag}.mpg ~/public_html/tmp/tmp

echo "--- Walltime: \${SECONDS} sec."
EOF
    (cd ${folder}; sbatch sub.slurm )
}


dz() {
  group="dz$1"
  cfl=$2
  zmax=10
  eps0=1.e-3
  anatime=1.0
  dumptime=10000000.0
  endtime=400.0
  ko=0.0
  nv=51
  dz=$1
  alpha=1.0
  ipt=1
sub ${group} ${cfl} ${zmax}  ${dz} ${nv} ${ipt} ${eps0} ${alpha} ${ko} ${anatime} ${dumptime} ${endtime}
}

anim() {
  alpha=$1
  ipt=$2
  group="anim"
  cfl=0.75
  zmax=1024
  eps0=1.e-4
  anatime=2.0
  dumptime=2.0
  endtime=3000.0
  ko=1e-3
  nv=129
  dz=0.1

  sub ${group} ${cfl} ${zmax}  ${dz} ${nv} ${ipt} ${eps0} ${alpha} ${ko} ${anatime} ${dumptime} ${endtime}
}

anim 0.9 0
#anim 1.1 0
#anim 0.9 1
#anim 1.1 0
#anim 1.1 1
