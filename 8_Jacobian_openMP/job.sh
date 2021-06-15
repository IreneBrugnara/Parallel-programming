#!/bin/bash
#SBATCH --job-name=openmp
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=32
#SBATCH --time=02:00:00
#SBATCH --partition=regular2
#SBATCH --output=log.txt
#SBATCH --hint=nomultithread


date

if [ "$1" = 'big' ]
then
  DIM=32000
  FREQ=100
else    # $2 = 'small'
  DIM=6400
  FREQ=10
fi

ITER=1000
NODES="1 2 4 8 16"


FILE="time_N"$1".txt"
> $FILE
echo $SLURM_JOB_ID
echo $SLURM_JOB_NODELIST
hostname

module load intel/19.1.3.304
module load openmpi3/3.1.4
module load phdf5/1.10.5
FLAGS="-I${HDF5_INC} -L${HDF5_LIB} -lhdf5"

mpicc $FLAGS -O3 -qopenmp jacobi_parallel.c -o jacobi_parallel.out -D PAR -D BLOCKING



for N in $NODES
do
  echo "running N=$N"
  
  # run the MPI only version
  P=$((32*$N))
  export OMP_NUM_THREADS=1
  OUTPUT=$(mpirun -np $P --map-by ppr:16:socket:pe=1 ./jacobi_parallel.out $DIM $ITER $FREQ)
  TCOMP=$(echo $OUTPUT | sed -n 1p)
  TCOMM=$(echo $OUTPUT | sed -n 2p)
  TDATA=$(echo $OUTPUT | sed -n 3p)
  echo -n "$N $TCOMP $TCOMM $TDATA" >> $FILE
  
  # run the hybrid MPI + openMP version
  P=$((2*$N))
  export OMP_NUM_THREADS=16
  OUTPUT=$(mpirun -np $P --map-by ppr:1:socket:pe=16 ./jacobi_parallel.out $DIM $ITER $FREQ)
  TCOMP=$(echo $OUTPUT | sed -n 1p)
  TCOMM=$(echo $OUTPUT | sed -n 2p)
  TDATA=$(echo $OUTPUT | sed -n 3p)
  echo "$TCOMP $TCOMM $TDATA" >> $FILE
done


date


