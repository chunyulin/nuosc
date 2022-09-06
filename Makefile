include Makefile.inc

TARGET= nuosc
OBJS  = nuosc.o nuosc_ana.o nuosc_class.o nuosc_init.o  nuosc_snapshot1D.o jacobi_poly.o

#MAP="/opt/arm/forge/21.1.2/bin/map --profile"
#ANA="/opt/arm/forge/21.1.2/bin/perf-report"

all: ${TARGET}

${OBJS}: nuosc_class.h

${TARGET}: ${OBJS}
	$(CXX) $(OPT) $(LIBS) $^ -o $@

test2d_gaussian:
	rm -f *.png *.bin  ~/public_html/tmp/tmp/ee*.png -f
	./nuosc --ipt 10 --nv 16 --dx 0.01 --xmax 1 --dz 0.01 --zmax 1 --mu 0 --pmo 0 --eps0 1 --sigma .2 --ko 1e-3 --cfl 0.5 --DUMP_EVERY_T 10 --END_STEP_T 50 --ANA_EVERY_T 1
	python3 ./plt_XZ.py ee*.bin
	scp *.png lincy@arm.nchc.org.tw:~/public_html/tmp/tmp/
test2d_square:
	rm -f *.png *.bin  ~/public_html/tmp/tmp/ee*.png -f
	./nuosc --ipt 20 --nv 8 --dx 0.01 --xmax 0.5 --dz 0.01 --zmax 0.8 --mu 0 --pmo 0 --eps0 1 --sigma .2 --ko 1e-3 --cfl 0.5 --DUMP_EVERY_T .2 --END_STEP_T 2 --ANA_EVERY_T .2
	python3 ./plt_XZ.py ee*.bin
	scp *.png lincy@arm.nchc.org.tw:~/public_html/tmp/tmp/

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
          ./nuosc --ipt 0 --pmo 0 --mu 1 --ko 1e-3 --zmax 1024 --dz 0.05 --nv 400 --cfl 0.5 --alpha 0.9 --eps0 1e-1 --sigma 2 --ANA_EVERY 100 --DUMP_EVERY_T 999 --END_STEP 500
	### |dP|_max ~ 1e-7 for at T~20

test2d:
	export CUDA_VISIBLE_DEVICES=1; \
          ./nuosc --pmo 0 --mu 1 --ko 1e-3 --ipt 0 --zmax 256 --xmax 0.5 --dz 0.1 --dx 0.1 --nv 33 --cfl 0.5 --alpha 0.9 --eps0 1e-1 --sigma 2 --ANA_EVERY 10 --DUMP_EVERY_T 999 --END_STEP 500
	### |dP|_max ~ 1e-7 for at T~20

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

