

#CXX=g++
#CXX=armclang++
CXX ?= armclang++

#CXX=nvc++

#INCLUDE=-I${ARMPL_INCLUDES}
INCLUDE=-I/pkg/gsl-2.7.1/include
LIBS=-L/pkg/gsl-2.7.1/lib -lgsl -lgslcblas
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
