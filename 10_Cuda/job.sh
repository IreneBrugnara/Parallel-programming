#!/bin/bash
#SBATCH --job-name=cuda_matmul
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --partition=gpu2
#SBATCH --output=log.txt
#SBATCH --hint=nomultithread
#SBATCH --mem-per-gpu=16G
#SBATCH --gpus-per-node=2


NODES="1 2 4"

FILE="time_N32000.txt"
> $FILE

DIM=32000
echo "dim = $DIM" >> $FILE

REPEAT=4
REPETITIONS=$(seq $REPEAT)
echo "repeat = $REPEAT" >> $FILE
echo "nodes   time   of which: comm." >> $FILE


# load modules
module load intel/19.1.3.304
module load openmpi3/3.1.4
module load cuda

# compile
nvcc -c matmul.c -O3 -I/opt/ohpc/pub/mpi/openmpi3-intel/3.1.4/include -Xcompiler -fopenmp

# link
nvcc -o matmul.x matmul.o -O3 -L/opt/ohpc/pub/mpi/openmpi3-intel/3.1.4/lib -lmpi -lcublas -lgomp

# run
for N in $NODES
do
  echo "running N=$N"
  AVE_TTOT=0
  AVE_TCOMM=0
  for R in $REPETITIONS
  do
    P=$((2*$N))
    export OMP_NUM_THREADS=16
    OUTPUT=$(mpirun -np $P --map-by ppr:1:socket:pe=16 ./matmul.x $DIM)
    TTOT=${OUTPUT% *}
    TCOMM=${OUTPUT#* }
    AVE_TTOT=$(echo "$TTOT+$AVE_TTOT" | bc)
    AVE_TCOMM=$(echo "$TCOMM+$AVE_TCOMM" | bc)
  done
  AVE_TTOT="$(echo "scale=9; $AVE_TTOT/$REPEAT" | bc)"
  AVE_TCOMM="$(echo "scale=9; $AVE_TCOMM/$REPEAT" | bc)"
  echo "$N $TTOT $TCOMM" >> $FILE
done

