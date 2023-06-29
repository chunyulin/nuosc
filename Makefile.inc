
MPI_ROOT?=
#GSL_ROOT?=/opt/ohpc/Taiwania3/libs/gsl/2.6
NCCL_ROOT?=/opt/nvidia/hpc_sdk/Linux_aarch64/22.2/comm_libs/nccl
CUDA_ROOT?=/opt/nvidia/hpc_sdk/Linux_aarch64/22.2/cuda

#CXX=nvc++
CXX ?= armclang++

#INCLUDE=-I${ARMPL_INCLUDES}
#INCLUDE=-I${GSL_ROOT}/include -I${MPI_ROOT}/include
INCLUDE=-I${MPI_ROOT}/include
# -I${NCCL_ROOT}/include

#LIBS=-Wl,-rpath -Wl,${GSL_ROOT}/lib -L${GSL_ROOT}/lib -lgsl -lgslcblas \
#     -Wl,-rpath -Wl,${MPI_ROOT}/lib -L${MPI_ROOT}/lib -lmpi
LIBS=-Wl,-rpath -Wl,${MPI_ROOT}/lib -L${MPI_ROOT}/lib -lmpi
# \
#     -L${CUDA_ROOT}/lib64  -lcudart \
#     -L${NCCL_ROOT}/lib  -lnccl \

##     -lnvToolsExt -lnccl 

## -lmpi_cxx   ## Intel2021 no need
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
