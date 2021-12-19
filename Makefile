include Makefile.inc

TARGET= nuosc
OBJS  = nuosc.o nuosc_ana.o nuosc_class.o

all: ${TARGET}

${OBJS}: nuosc_class.h

${TARGET}: ${OBJS}
	$(CXX) $(OPT) $^ -o $@ $(LIBS)

run:
	#numactl -C 0-15 
	./${TARGET} --ANA_EVERY 4 --ENDSTEP 400

clean:
	rm *.o -f ${TARGET}

