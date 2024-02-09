include Makefile.inc

TARGET= nuosc
OBJS  = nuosc.o nuosc_class.o nuosc_rhs.o nuosc_init.o nuosc_boundary.o nuosc_sync.o nuosc_ana.o nuosc_snapshot.o
## jacobi_poly.o
#MAP="/opt/arm/forge/21.1.2/bin/map --profile"
#ANA="/opt/arm/forge/21.1.2/bin/perf-report"

all: ${TARGET}

${OBJS}: nuosc_class.h common.h

${TARGET}: ${OBJS}
	$(CXX) $(OPT) $(LIBS) $^ -o $@

test3d:
	mpirun -np 1 ./nuosc --np 1 1 1 --pmo 0 --mu 1 --ko 1e-3 --ipt 0 --xmax 5 .2 .25 --dx 0.1 --nv 8 --nphi 8 --cfl 0.5 --alpha 0.9 --eps0 1e-1 --sigma 1 --ANA_EVERY 5 --END_STEP 10

test3d_gaussian:
	#rm -f *.png *.bin  ~/public_html/tmp/tmp/ee*.png -f
	mpirun -np 1 ./nuosc --np 1 1 1 --ipt 10 --nv 4 --dx 0.05 --xmax 1.5 .1 1.5 --mu 0 --pmo 0 --eps0 1 --sigma 10 --ko 1e-3 --cfl 0.1 --END_STEP 10 --ANA_EVERY 1 --DUMP_EVERY 2
	#python3 ./plt_ZX.py ee*.bin
	#scp *.png lincy@arm.nchc.org.tw:~/public_html/tmp/tmp/

test2d_gaussian:
	#rm -f *.png *.bin  ~/public_html/tmp/tmp/ee*.png -f
	./nuosc --ipt 10 --nv 5 --dx 0.01 --xmax 1 --dz 0.01 --zmax 1 --mu 0 --pmo 0 --eps0 1 --sigma .2 --ko 1e-3 --cfl 0.5 --DUMP_EVERY_T 10 --END_STEP_T 50 --ANA_EVERY_T 1
	#python3 ./plt_XZ.py ee*.bin
	#scp *.png lincy@arm.nchc.org.tw:~/public_html/tmp/tmp/

test_gaussian:
	rm -f *.png *.bin  ~/public_html/tmp/tmp/ee*.png -f
	./nuosc --ipt 10 --nv 3 --dz 0.02 --zmax 0.5 --mu 0 --eps0 1 --sigma .1 --cfl 0.5 --DUMP_EVERY_T 5 --END_STEP_T 50 --ANA_EVERY_T 5
	python3 ./plt_overlay.py ee*.bin
	cp -f *.png ~/public_html/tmp/tmp/

test_square:
	rm -f *.png *.bin  ~/public_html/tmp/tmp/ee*.png -f
	./nuosc --ipt 20 --nv 3 --dz 0.001 --zmax 0.5 --mu 0 --eps0 1 --sigma .1 --cfl 0.5 --ko 4 --DUMP_EVERY_T 0.25 --END_STEP_T 1 --ANA_EVERY_T 1
	python3 ./plt_overlay.py ee*.bin
	cp -f *.png ~/public_html/tmp/tmp/

test_tri:
	rm -f *.png *.bin  ~/public_html/tmp/tmp/ee*.png -f
	./nuosc --ipt 30 --nv 3 --dz 0.001 --zmax 0.5 --mu 0 --eps0 1 --sigma .1 --cfl 0.5 --DUMP_EVERY_T 5 --END_STEP_T 50 --ANA_EVERY_T 10
	python3 ./plt_overlay.py ee*.bin
	cp -f *.png ~/public_html/tmp/tmp/

test:
	export CUDA_VISIBLE_DEVICES=0; \
          ./nuosc --pmo 0 --mu 1 --ko 1e-3 --ipt 0 --zmax 512 --dz 0.1                     --nv 33 --cfl 0.5 --alpha 0.9 --eps0 1e-1 --sigma 5 --ANA_EVERY_T 10 --DUMP_EVERY_T 9999999 --END_STEP_T 1000
	### |dP|_max ~ 1e-7 for at T~20

test2d:
	export CUDA_VISIBLE_DEVICES=0; \
          ./nuosc --pmo 0 --mu 1 --ko 1e-3 --ipt 0 --zmax 512 --xmax 0.5 --dz 0.1 --dx 0.1 --nv 33 --nphi 8 --cfl 0.5 --alpha 0.9 --eps0 1e-1 --sigma 5 --ANA_EVERY_T 10 --DUMP_EVERY_T 10000 --END_STEP_T 3000
	### |dP|_max ~ 1e-7 for at T~20

PARAM3D=--pmo 0 --mu 1 --ko 1e-3 --ipt 0 --xmax 2 .2 2 --dx 0.1 --nv 7 --nphi 8 --cfl 0.5 --alpha 0.9 --eps0 1e-1 --sigma 1 --ANA_EVERY 2 --END_STEP 10
test3d_mpi:
	mpirun -np 1 ./nuosc --np 1 1 1 ${PARAM3D}
	mpirun -np 4 ./nuosc --np 2 1 2 ${PARAM3D}
	mpirun -np 4 ./nuosc --np 1 2 2 ${PARAM3D}
	mpirun -np 2 ./nuosc --np 2 1 1 ${PARAM3D}


PARAM=--pmo 0 --mu 1 --ko 1e-3 --ipt 0 --zmax 32 --xmax 32 --dz 0.1 --dx 0.1 \
       --nv 17 --nphi 8 --cfl 0.5 --alpha 0.9 --eps0 1e-1 --sigma 5 --ANA_EVERY 5 --END_STEP 10
test2d_mpi:
	mpirun -np 1 ./nuosc --np 1 1 ${PARAM}
	mpirun -np 2 ./nuosc --np 2 1 ${PARAM}
	mpirun -np 2 ./nuosc --np 1 2 ${PARAM}
	mpirun -np 4 ./nuosc --np 2 2 ${PARAM}
	mpirun -np 4 ./nuosc --np 4 1 ${PARAM}
	mpirun -np 4 ./nuosc --np 1 4 ${PARAM}
	mpirun -np 8 ./nuosc --np 4 2 ${PARAM}
	mpirun -np 8 ./nuosc --np 2 4 ${PARAM}

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


test_vint: test_vint.o jacobi_poly.o
	nvc++ test_vint.cpp jacobi_poly.cpp -I/pkg/gsl-2.7.1/include -L/pkg/gsl-2.7.1/lib -lgsl -lgslcblas -o test_vint


clean:
	rm *.o -f ${TARGET}

