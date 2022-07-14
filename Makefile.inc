

#CXX=g++
#CXX=armclang++
#CXX=nvc++

#INCLUDE=-I/pkg/gcc9/papi-6.0/include
#LIBS=-L/pkg/gcc9/papi-6.0/lib -lpapi
#DEBUG=-DPAPI
#OPT=-Ofast -std=c++11 -fopenmp

#OPT=-mp -O3 -Minfo=mp

#OPT=-acc=gpu -O3 -Minfo=acc -ta=tesla:managed
#CFLAGS=-DNVTX
#LIBS=-lnvToolsExt

LIBS=-L/pkg/armclang/gsl-2.7.1/lib -lgsl -lgslcblas


## -fp-model precise -unroll-aggressive -parallel -opt-prefetch 
.cpp.o:
	$(CXX) $(CFLAGS) $(OPT) $(INCLUDE) $< -c -o $@ $(DEBUG)
