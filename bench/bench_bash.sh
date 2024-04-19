#!/bin/bash
sub1d() {
    local fname=`basename $BASH_SOURCE`;
    local runtag=$1; shift

    local gpu=$1; shift

    local zm=$1; shift
    local dx=$1; shift
    local nv=$1; shift

    local sigma=$1; shift    ### slightly change the onset of depolarization
    local eps=$1; shift      ### larger eps eariler growth,

    local ET=$1; shift
    local AT=$1; shift
    local DT=$1; shift

    printf -v v0 "%03d" $nv
    local jobtag=v${nv}
    local folder=${runtag}/${jobtag}
    mkdir ${folder} -p

    cp ../nuosc ${fname} ${folder}

    local subfile=${jobtag}.sh
    cat << EOF > ${folder}/${subfile}
ml purge
ml nvhpc
export CUDA_VISIBLE_DEVICES=${gpu}
 ./nuosc --zmax ${zm} --dz ${dx} --nv ${nv} --cfl 0.5 --pmo 0 --mu 1 --ko 1e-3 \
         --ipt 0 --alpha 0.9 --lnue 0.6 --lnueb 0.5 --eps0 ${eps} --sigma ${sigma} \
         --ANA_EVERY_T ${AT} --DUMP_EVERY_T ${DT} --END_STEP_T ${ET}

  echo "---- Walltime: \${SECONDS} sec."
EOF
    #(cd ${folder}; nohup bash ${subfile} &  )
    (cd ${folder}; bash ${subfile}  )
}

sub1d fZ10s5e2  0   10  0.1   64      5  1e-2       1000 2 9999 &
sub1d fZ10s5e3  0   10  0.1   64      5  1e-3       1000 2 9999 &
sub1d fZ10s5e5  1   10  0.1   64      5  1e-5       1000 2 9999

