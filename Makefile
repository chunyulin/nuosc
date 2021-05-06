include Makefile.inc

TARGET= nuosc
OBJS  = nuosc.o

${TARGET}: nuosc.o
	$(CXX) $(OPT) $< -o $@ $(LIBS)

run:
	numactl -C 0-127 ./${TARGET} --mu 1.0 --renorm 1 --nvz 51 --dz 0.4 --zmax 600 --cfl 0.4 --nvz 51 --eps0 0.1 --ana_per 50 --endstep 5000

test:
	./${TARGET} --mu 10; mv rate.dat rate_10.dat
	./${TARGET} --mu 50; mv rate.dat rate_50.dat
	./${TARGET} --mu 100; mv rate.dat rate_100.dat

clean:
	rm *.o -f ${TARGET}

