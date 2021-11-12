

CXX=g++
CXX=nvc++

#INCLUDE=-I/pkg/gcc9/papi-6.0/include
#LIBS=-L/pkg/gcc9/papi-6.0/lib -lpapi
#DEBUG=-DPAPI

OPT=-O3 -std=c++11 -fopenmp
OPT=-acc=gpu -mp -O3 -Minfo=mp
OPT=-acc=gpu -O3 -Minfo=acc -ta=tesla:managed


CFLAGS=-DNVTX
LIBS=-lnvToolsExt

## -fp-model precise 
#-unroll-aggressive -parallel -opt-prefetch 

.cpp.o:
	$(CXX) $(CFLAGS) $(OPT) $(INCLUDE) $< -c -o $@ $(DEBUG)
