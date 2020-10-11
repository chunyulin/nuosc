include Makefile.inc

TARGET= nuosc
OBJS  = nuosc.o

${TARGET}: nuosc.o
	$(CXX) $(OPT) $< -o $@ $(LIBS)

clean:
	rm *.o -f ${TARGET}

