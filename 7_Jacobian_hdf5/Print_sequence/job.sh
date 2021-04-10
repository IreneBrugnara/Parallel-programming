#!/bin/bash
#SBATCH --job-name=hdf5
#SBATCH --ntasks-per-node=20
#SBATCH --time=00:50:00
#SBATCH --partition=regular1
#SBATCH --output=log.txt
#SBATCH --hint=nomultithread


# expected first argument: "-N 1" or "-N 2"
# expected second argument: 1200 or 12000
# expected third argument: "par" or "ser"
# expected fourth argument: number of iterations
# expected fifth argument: dump frequency (integer)


DIM=$1  #1200 or 12000
ITER=$3
NODES=$SLURM_JOB_NUM_NODES
P=$((20*$NODES))
FREQ=$4

echo $ITER
echo $FREQ

module load intel/19.1.3.304
module load openmpi3/3.1.4
module load phdf5/1.10.5
FLAGS="-I${HDF5_INC} -L${HDF5_LIB} -lhdf5"

if [ "$2" = 'par' ]
then
  PARAM="-D PAR"
else
  PARAM=""
fi

mpicc $FLAGS jacobi_parallel.c -O3 -o jacobi_parallel.out $PARAM

mpirun -np $P ./jacobi_parallel.out $DIM $ITER $FREQ



