include Makefile.inc

TARGET= nuosc
OBJS  = nuosc.o nuosc_ana.o nuosc_class.o nuosc_init.o  nuosc_snapshot.o

#MAP="/opt/arm/forge/21.1.2/bin/map --profile"
#ANA="/opt/arm/forge/21.1.2/bin/perf-report"

all: ${TARGET}

${OBJS}: nuosc_class.h

${TARGET}: ${OBJS}
	$(CXX) $(OPT) $^ -o $@ $(LIBS)

test_gaussian:
	rm -f *.png *.bin  ~/public_html/tmp/tmp/ee*.png -f
	./nuosc --ipt 10 --nv 3 --dz 0.02 --zmax 0.5 --mu 0 --eps0 1 --sigma .1 --cfl 0.5 --DUMP_EVERY_T 5 --END_STEP_T 50 --ANA_EVERY_T 5
	python3 ./plt_overlay.py ee*.bin
	cp -f *.png ~/public_html/tmp/tmp/

test_square:
	rm -f *.png *.bin  ~/public_html/tmp/tmp/ee*.png -f
	./nuosc --ipt 20 --nv 3 --dz 0.001 --zmax 0.5 --mu 0 --eps0 1 --sigma .1 --cfl 0.5 --DUMP_EVERY_T 0.25 --END_STEP_T 1 --ANA_EVERY_T 1
	python3 ./plt_overlay.py ee*.bin
	cp -f *.png ~/public_html/tmp/tmp/

test_tri:
	rm -f *.png *.bin  ~/public_html/tmp/tmp/ee*.png -f
	./nuosc --ipt 30 --nv 3 --dz 0.001 --zmax 0.5 --mu 0 --eps0 1 --sigma .1 --cfl 0.5 --DUMP_EVERY_T 5 --END_STEP_T 50 --ANA_EVERY_T 10
	python3 ./plt_overlay.py ee*.bin
	cp -f *.png ~/public_html/tmp/tmp/

run:
	#numactl -C 0-15 
	#./${TARGET} --ipt 4 --zmax 5120 --dz 0.5 --nv 201 --ko 0.1 --cfl 0.4 --ANA_EVERY_T 1 --DUMP_EVERY_T 999 --END_STEP_T 10
	./nuosc --ipt 1 --zmax 16 --dz 0.1 --nv 101 --ko 0.1 --cfl 0.5 --eps0 0.001 --sigma 5 --ANA_EVERY 1 --DUMP_EVERY_T 999 --END_STEP 40
	
nsys:
	NSYS=/work/opt/ohpc/pkg/qchem/nv/nsight-systems-2020.3.1/bin/nsys
	${NSYS} profile -o nuosc.%q{PMIX_RANK} -f true -t openmp ./nuosc

clean:
	rm *.o -f ${TARGET}

