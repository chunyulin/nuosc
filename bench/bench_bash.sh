#!/bin/bash
sub_shell() {
    local fname=`basename $BASH_SOURCE`;
    local runtag=$1; shift

    local gpu=$1; shift

    local xm=$1; shift
    local ym=$1; shift
    local zm=$1; shift
    local dx=$1; shift

    local nv=$1; shift
    local nphi=$1; shift

    local ipt=$1; shift
    local nu=$1; shift
    local nub=$1; shift
    local alpha=$1; shift
    local sigma=$1; shift
    local eps=$1; shift

    local ET=$1; shift
    local AT=$1; shift
    local DT=$1; shift


    printf -v g0 "%d" $gpu
    printf -v v0 "%03d" $nv
    printf -v p0 "%02d" $nphi
    local jobtag=v${v0}p${p0}
    local folder=${runtag}/${jobtag}
    mkdir ${folder} -p

    cp ../nuosc ${fname} ${folder}

    local subfile=${jobtag}.sh
    cat << EOF > ${folder}/${subfile}
ml purge
ml nvhpc
export CUDA_VISIBLE_DEVICES=${gpu}
mpirun -np 1 \
 ./nuosc --np 1 1 1 --xmax ${xm} ${ym} ${zm} --dx ${dx} --nv ${nv} --nphi ${nphi} --cfl 0.5 --pmo 0 --mu 1 --ko 1e-3 \
         --ipt ${ipt} --alpha ${alpha} --${nu} 0.6 --${nub} 0.5 --eps0 ${eps} --sigma ${sigma} --restart -1 \
         --ANA_EVERY_T ${AT} --DUMP_EVERY_T ${DT} --END_STEP_T ${ET}

  echo "---- Walltime: \${SECONDS} sec."
EOF
    (cd ${folder}; nohup bash ${subfile} &  )
    #(cd ${folder}; bash ${subfile}  )
}

#sub_shell test 0   .2 0.2 0.2   0.1    16 99     2 lnue lnueb 0.9 5 1e-6   1 2 9999
sub_shell fX10 0   10 0.3 0.3   0.1   10 99     0 lnuex lnuebx 0.9 5 1e-3   1000 2 9999
sub_shell fZ10 1   0.3 0.3 10   0.1   10 99     2 lnue  lnueb  0.9 5 1e-3   1000 2 9999
#sub_shell fY10 0   0.3 10  0.3  0.1    8 99     1 lnuey lnueby 0.9 5 1e-3   1000 2 9999


#sub_shell fX60 1   60 0.3 0.3   0.1    5 99     0 lnuex lnuebx 0.9 5 1e-6   1000 2 9999
#sub_shell fZ60 1   0.3 0.3 60   0.1    5 99     2 lnue  lnueb  0.9 5 1e-6   1000 2 9999
#sub_shell fY60 1   0.3 60  0.3  0.1    5 99     1 lnuey lnueby 0.9 5 1e-6   1000 2 9999

#sub_shell fX15 0   15 0.3 0.3   0.1    8 99     0 lnuex lnuebx 0.9 5 1e-6   1000 2 9999
#sub_shell fZ15 0   0.3 0.3 15   0.1    8 99     2 lnue  lnueb  0.9 5 1e-6   1000 2 9999
#sub_shell fY15 0   0.3 15  0.3  0.1    8 99     1 lnuey lnueby 0.9 5 1e-6   1000 2 9999

#sub_shell fX6 1   6   0.3 0.3  0.1    16 32     0 lnuex lnuebx 0.9 1 1e-6   600 2 9999
#sub_shell fZ6 1   0.3 0.3 6    0.1    16 32     2 lnue  lnueb  0.9 1 1e-6   600 2 9999
#sub_shell fY6 1   0.3 6   0.3  0.1    16 32     1 lnuey lnueby 0.9 1 1e-6   600 2 9999
#sub_shell fX6 1   6   0.3 0.3  0.1    16 16     0 lnuex lnuebx 0.9 1 1e-6   600 2 9999
#sub_shell fZ6 1   0.3 0.3 6    0.1    16 16     2 lnue  lnueb  0.9 1 1e-6   600 2 9999
#sub_shell fY6 1   0.3 6   0.3  0.1    16 16     1 lnuey lnueby 0.9 1 1e-6   600 2 9999
