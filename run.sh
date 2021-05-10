#!/bin/bash

sub() {
    local runtag=$1; shift
    local cfl=$1; shift
    local zmax=$1; shift
    local dz=$1; shift
    local nvz=$1; shift
    local renorm=$1; shift
    local eps0=$1; shift
    local alpha=$1; shift
    local ko=$1; shift
    local ana=$1; shift
    local dump=$1; shift
    local end=$1; shift

    folder="${runtag}/${runtag}_r${renorm}e${eps0}v${nvz}z${dz}k${ko}"
    mkdir ${folder} -p
    cp ./nuosc ${runtag}
    
    cat << EOF > ${folder}/sub.slurm
#!/bin/bash
#SBATCH --job-name ${runtag}_${renorm}_${dz}_${eps0}
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 60
#SBATCH --mem-bind=verbose,p
module purge
module load ThunderX2CN99/RHEL/7/gcc-9.3.0/armpl
srun --cpu-bind=v,cores \
  ../nuosc --mu 1.0 --nvz ${nvz} --dz ${dz} --zmax ${zmax} --cfl ${cfl} \\
  --renorm ${renorm} --eps0 ${eps0} --alpha ${alpha} --ko ${ko} \\
  --ANA_EVERY_T ${ana} --DUMP_EVERY_T ${dump} --ENDSTEP_T ${end}
echo "--- Walltime: \${SECONDS} sec."
EOF
    (cd ${folder}; sbatch sub.slurm )
}


subG3lin () {
group="G3lin"
cfl=0.4
zmax=600
eps0=1.e-6
alpha=0.9
anatime=4.0
dumptime=100.0
endtime=500.0

sub ${group} ${cfl} ${zmax}  0.4 21  0  ${eps0} ${alpha} 0.0    ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.0    ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.01   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.001  ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 21  1  ${eps0} ${alpha} 0.0    ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.0    ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.01   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.001  ${anatime} ${dumptime} ${endtime}
}

subG4lin () {
group="G4lin"
cfl=0.4
zmax=600
eps0=1.e-7
alpha=0.92
anatime=4.0
dumptime=100.0
endtime=500.0

sub ${group} ${cfl} ${zmax}  0.4 21  0  ${eps0} ${alpha} 0.0   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.0   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.01  ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.001 ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 21  1  ${eps0} ${alpha} 0.0   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.0   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.01  ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.001 ${anatime} ${dumptime} ${endtime}
}

subG3 () {
group="G3"
cfl=0.4
zmax=1200
eps0=0.1
alpha=0.9
anatime=4.0
dumptime=100
endtime=1200.0

sub ${group} ${cfl} ${zmax}  0.4 21  0  ${eps0} ${alpha} 0.0    ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.0    ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.01   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.001  ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 21  1  ${eps0} ${alpha} 0.0    ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.0    ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.01   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.001  ${anatime} ${dumptime} ${endtime}
}

subG3 () {
group="G4"
cfl=0.4
zmax=1200
eps0=0.1
alpha=0.92
anatime=4.0
dumptime=100
endtime=1200.0

sub ${group} ${cfl} ${zmax}  0.4 21  0  ${eps0} ${alpha} 0.0   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.0   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.01  ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.001 ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 21  1  ${eps0} ${alpha} 0.0   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.0   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.01  ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.001 ${anatime} ${dumptime} ${endtime}
}

group="G3long"
cfl=0.4
zmax=1200
eps0=0.1
alpha=0.92
anatime=4.0
dumptime=200
endtime=3000.0

sub ${group} ${cfl} ${zmax}  0.4 21  0  ${eps0} ${alpha} 0.0   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.0   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.01  ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  0  ${eps0} ${alpha} 0.001 ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 21  1  ${eps0} ${alpha} 0.0   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.0   ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.01  ${anatime} ${dumptime} ${endtime}
sub ${group} ${cfl} ${zmax}  0.4 41  1  ${eps0} ${alpha} 0.001 ${anatime} ${dumptime} ${endtime}
