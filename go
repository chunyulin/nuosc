NSYS=/work/opt/ohpc/pkg/qchem/nv/nsight-systems-2020.3.1/bin/nsys


${NSYS} profile -o nuosc.%q{PMIX_RANK} -f true -t openmp ./nuosc
