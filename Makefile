include Makefile.inc

TARGET= nuosc
OBJS  = nuosc.o nuosc_ana.o nuosc_class.o

all: ${TARGET}

${OBJS}: nuosc_class.h

${TARGET}: ${OBJS}
	$(CXX) $(OPT) $^ -o $@ $(LIBS)

run:
	#numactl -C 0-15 
	./${TARGET} --zmax 20 --dz 0.1 --nv 101 --ko 0.1 --cfl 0.4 --ANA_EVERY_T 2 --DUMP_EVERY_T 2 --ENDSTEP_T 40

nsys:
	NSYS=/work/opt/ohpc/pkg/qchem/nv/nsight-systems-2020.3.1/bin/nsys
	${NSYS} profile -o nuosc.%q{PMIX_RANK} -f true -t openmp ./nuosc

clean:
	rm *.o -f ${TARGET}

