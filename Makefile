include Makefile.inc

TARGET= nuosc
OBJS  = nuosc.o

${TARGET}: nuosc.o
	$(CXX) $(OPT) $< -o $@ $(LIBS)
aaa: aaa.o
	$(CXX) $(OPT) $< -o $@ $(LIBS)

run:
	numactl -C 0-127 ./${TARGET}

test:
	./${TARGET} --mu 10; mv rate.dat rate_10.dat
	./${TARGET} --mu 50; mv rate.dat rate_50.dat
	./${TARGET} --mu 100; mv rate.dat rate_100.dat

clean:
	rm *.o -f ${TARGET}

