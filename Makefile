include Makefile.inc

TARGET= nuosc
OBJS  = nuosc.o

${TARGET}: nuosc.o
	$(CXX) $(OPT) $< -o $@ $(LIBS)

run:
	numactl -C 0-127 ./${TARGET} --mu 1.0 --renorm 1 --nvz 51 --dz 0.4 --zmax 600 --cfl 0.4 --eps0 0.1 --alpha 0.9 --ANA_EVERY_T 4.0 --ENDSTEP_T 16.0

clean:
	rm *.o -f ${TARGET}

