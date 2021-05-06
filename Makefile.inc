

CXX=g++

INCLUDE=-I/pkg/gcc9/papi-6.0/include
#LIBS=-L/pkg/gcc9/papi-6.0/lib -lpapi
#DEBUG=-DPAPI

OPT=-O3 -std=c++11 -fopenmp


## -fp-model precise 
#-unroll-aggressive -parallel -opt-prefetch 

.cpp.o:
	$(CXX) $(OPT) $(INCLUDE) $< -c -o $@ $(DEBUG)
