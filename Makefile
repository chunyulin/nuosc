include Makefile.inc

TARGET= nuosc
OBJS  = nuosc.o

${TARGET}: nuosc.o
	$(CXX) $(OPT) $< -o $@ $(LIBS)

run:
	numactl -C 0-127 ./${TARGET}

clean:
	rm *.o -f ${TARGET}

