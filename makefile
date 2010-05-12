########################################################################
# Makefile for MPI Homework
#
########################################################################

CC         =  mpic++
CCFLAGS    =  -fast -xtarget=native -xunroll=100
LIBS       =  -lmpi

all:
	@echo "Usage: make hello"

fox:   mpi_fox.cpp
	$(CC) $(CCFLAGS) -o fox mpi_fox.cpp $(LIBS)

clean: 
	rm *.o	
		
