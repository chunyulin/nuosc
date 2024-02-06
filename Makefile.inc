
MPI_ROOT?=
GSL_ROOT?=/opt/ohpc/Taiwania3/libs/gsl/2.6

#CXX=g++
#CXX=armclang++

#INCLUDE=-I${ARMPL_INCLUDES}
INCLUDE=-I${MPI_ROOT}/include ##-I${GSL_ROOT}/include 

LIBS?=-Wl,-rpath -Wl,${MPI_ROOT}/lib -L${MPI_ROOT}/lib -lmpi \

#     -lnvToolsExt
#-Wl,-rpath -Wl,${GSL_ROOT}/lib -L${GSL_ROOT}/lib -lgsl -lgslcblas \

## -lmpi_cxx   ## Intel2021 no need
#DEBUG=-DPAPI
#OPT=-Ofast -std=c++11 -fopenmp ${COSENU_OPT}
OPT?= -Ofast -std=c++11 -fopenmp -armpl

.SUFFIXES : .cpp .o .cu

## -fp-model precise -unroll-aggressive -parallel -opt-prefetch 
.cpp.o:
	$(CXX) $(CFLAGS) $(OPT) $(INCLUDE) $< -c -o $@ $(DEBUG)

.cu.o:
	$(CXX) $(CFLAGS) $(OPT) $(INCLUDE) $< -c -o $@ $(DEBUG)
