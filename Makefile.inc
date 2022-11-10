
MPI_ROOT?=/opt/nvidia/hpc_sdk/Linux_aarch64/22.2/comm_libs/openmpi/openmpi-3.1.5
GSL_ROOT?=/opt/ohpc/pkg/kagra/nv/gsl-2.7.1
  
#CXX=g++
#CXX=armclang++
CXX ?= armclang++

#CXX=nvc++

#INCLUDE=-I${ARMPL_INCLUDES}
INCLUDE=-I${GSL_ROOT}/include -I${MPI_ROOT}/include

LIBS=-Wl,-rpath -Wl,${GSL_ROOT}/lib -L${GSL_ROOT}/lib -lgsl -lgslcblas \
     -Wl,-rpath -Wl,${MPI_ROOT}/lib -L${MPI_ROOT}/lib -lmpi

#DEBUG=-DPAPI
#OPT=-Ofast -std=c++11 -fopenmp ${COSENU_OPT}
OPT ?= -Ofast -std=c++11 -fopenmp -armpl
#OPT=-mp -O3 -Minfo=mp

#OPT=-acc=gpu -O3 -Minfo=acc -ta=tesla:managed
#CFLAGS ?= 
#LIBS ?= -lnvToolsExt


.SUFFIXES : .cpp .o .cu

## -fp-model precise -unroll-aggressive -parallel -opt-prefetch 
.cpp.o:
	$(CXX) $(CFLAGS) $(OPT) $(INCLUDE) $< -c -o $@ $(DEBUG)

.cu.o:
	$(CXX) $(CFLAGS) $(OPT) $(INCLUDE) $< -c -o $@ $(DEBUG)
