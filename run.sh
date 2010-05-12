#!/it/sw/misc/bin/bash

#
# BASH script to run the MPI programs
# 1st arg : number of Processors , 2nd arg : Order of Matrices
#
if [ "$1" -le "4" ]; then
	echo "Running on a single machine"
	mpirun -np $1 fox -o $2
else
	echo "Running on a cluster now"
	mpirun -hostfile nodes -mca plm_rsh_agent rsh -np $1 fox -o $2
fi

