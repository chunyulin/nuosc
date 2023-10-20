include Makefile.inc

TARGET= nuosc
OBJS  = nuosc.o nuosc_ana.o nuosc_class.o nuosc_init.o nuosc_boundary.o  CartGrid.o  nuosc_snapshot.o
##\\ jacobi_poly.o 

#MAP="/opt/arm/forge/21.1.2/bin/map --profile"
#ANA="/opt/arm/forge/21.1.2/bin/perf-report"

all: ${TARGET}

${OBJS}: nuosc_class.h CartGrid.h common.h

${TARGET}: ${OBJS}
	$(CXX) $(OPT) $(LIBS) $^ -o $@

test2d:
	export OMP_NUM_THREADS=40; \
	mpirun -np 1 ./nuosc --np 1 1 --pmo 0 --mu 1 --ko 1e-3 --ipt 0 --zmax 50 --xmax 0.2 --dz 0.1 --dx 0.1 --nv 8 --nphi 8 --cfl 0.5 --alpha 0.9 --eps0 1e-1 --sigma 1 \
        --ANA_EVERY 5 --END_STEP 20

UNITTEST=--nv 4 --dx 0.01 --xmax 0.5 --dz 0.01 --zmax 0.5 --mu 0 --pmo 0 --eps0 1 --sigma .2 --ko 1e-3 --cfl 0.5 --DUMP_EVERY_T .1 --END_STEP_T .5 --ANA_EVERY_T 0.1
UNITTEST=--nv 2 --gz 2 --dz 0.02  --zmax 0.5 --mu 0 --pmo 0 --eps0 1 --sigma .2 --ko 0.8 --cfl 0.25 --DUMP_EVERY_T .2 --END_STEP_T .5 --ANA_EVERY_T .2
UNITTEST=--nv 2 --gz 2 --dz 0.002 --zmax 0.5  --xmax 0.004  --mu 0 --pmo 0 --eps0 1 --sigma .08 --ko 0.0 --cfl 0.5 --DUMP_EVERY_T 1 --END_STEP_T 5 --ANA_EVERY_T 2
test2d_advec:
	rm -f *.png *.bin -f
	./nuosc --ipt 10 ${UNITTEST}
	python3 ./plt_overlay.py ee*.bin
	#python3 ./plt_ZX.py ee*.bin

test:
	export CUDA_VISIBLE_DEVICES=0; \
          ./nuosc --pmo 0 --mu 1 --ko 1e-3 --ipt 0 --zmax 512 --dz 0.1                     --nv 33 --cfl 0.5 --alpha 0.9 --eps0 1e-1 --sigma 5 --ANA_EVERY_T 10 --DUMP_EVERY_T 9999999 --END_STEP_T 1000
	### |dP|_max ~ 1e-7 for at T~20

PARAM=--pmo 0 --mu 1 --ko 1e-3 --ipt 0 --zmax 2 --xmax 2 --dz 0.1 --dx 0.1 \
       --nv 7 --nphi 8 --cfl 0.5 --alpha 0.9 --eps0 1e-1 --sigma 1 --ANA_EVERY 2 --END_STEP 10
test2d_mpi:
	mpirun -np 1 ./nuosc --np 1 1 ${PARAM}
	mpirun -np 2 ./nuosc --np 1 2 ${PARAM}

testmpi: testmpi.o
	mpic++ -acc -Minfo=acc testmpi.cpp -o testmpi
	
test_fft: test_fft.o
	$(CXX) $(OPT) $^ -o $@ $(LIBS)
	
test_cufft: test_cufft.o
	$(CXX) $(OPT) $^ -o $@ $(LIBS)
	
NSYS=/work/opt/ohpc/pkg/qchem/nv/nsight-systems-2020.3.1/bin/nsys
NSYS=nsys
NCU=/opt/nvidia/hpc_sdk/Linux_aarch64/22.2/compilers/bin/ncu
nsys:
	export CUDA_VISIBLE_DEVICES=0; \
           ${NSYS} profile -o nuosc -f true -t cuda,nvtx \
           ./nuosc --ipt 0 --pmo 1e-5 --mu 1 --ko 1e-3 --zmax 1024 --dz 0.05 --nv 400 --cfl 0.5 --alpha 0.9 --eps0 1e-1 --sigma 2 \
              --ANA_EVERY 999 --DUMP_EVERY 999 --END_STEP 10
ncu:
	${NCU} --nvtx -f --set full --sampling-interval 6 -o FD \
           ./nuosc --ipt 0 --mu 1.0 --zmax 1024 --dz 0.05 --nv 400 --cfl 0.5 --ko 1e-3 --ANA_EVERY 9999 --DUMP_EVERY 9999 --END_STEP 5

clean:
	rm *.o -f ${TARGET}

