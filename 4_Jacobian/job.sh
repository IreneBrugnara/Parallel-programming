#!/bin/bash
#SBATCH --job-name=jacobi
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=20
#SBATCH --time=00:10:00
#SBATCH --partition=regular1
#SBATCH --output=log.txt
#SBATCH --hint=nomultithread

# expected first argument: 1200 or 12000
# expected second argument: "blocking" or "nonblocking"


date
DIM=$1  #1200 or 12000
ITER=10
FILE="time_N$1_$2.dat"
INFO="# N=$1 $2"
echo $INFO > $FILE
echo $SLURM_JOB_ID
echo $SLURM_JOB_NODELIST

LIST_P='20 40 80 160'

if [ "$2" = 'blocking' ]
then
  PARAM="-D BLOCKING"
else
  PARAM=""
fi

hostname
module load gnu8/8.3.0
module load openmpi3
mpicc jacobi_parallel.c -O3 -o jacobi_parallel.out $PARAM # -D PRINT -D DEBUG


AVE_COMM=0
AVE_COMP=0
REPEAT=10
REPETITIONS=$(seq $REPEAT)


for P in $LIST_P
do
  echo "running P=$P"
  for R in $REPETITIONS
  do
    OUTPUT=$(mpirun -np $P ./jacobi_parallel.out $DIM $ITER)
    COMM=${OUTPUT% *}
    COMP=${OUTPUT#* }
    AVE_COMM=$(echo "$COMM+$AVE_COMM" | bc)
    AVE_COMP=$(echo "$COMP+$AVE_COMP" | bc)
  done
  echo $AVE_COMM
  AVE_COMM="$(echo "scale=5; $AVE_COMM/$REPEAT" | bc)"
  echo $AVE_COMM
  AVE_COMP=$(echo "scale=5; $AVE_COMP/$REPEAT" | bc)
  echo "$P $AVE_COMM $AVE_COMP" >> $FILE
done


date


