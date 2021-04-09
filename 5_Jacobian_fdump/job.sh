#!/bin/bash
#SBATCH --job-name=jacobi
#SBATCH --ntasks-per-node=20
#SBATCH --time=00:10:00
#SBATCH --partition=regular1
#SBATCH --output=log.txt
#SBATCH --hint=nomultithread

# expected argument for sbatch: "-N 1" or "-N 2"
# expected first argument: 1200 or 12000
# expected second argument: "text" or "binary"
# example: sbatch -N 2 job.sh 1200 binary

date
DIM=$1  #1200 or 12000
ITER=10
NODES=$SLURM_JOB_NUM_NODES
P=$((20*$NODES))
RATIO=10

FILE="time_N"$1"_"$2"_"$NODES"nodes.txt"
> $FILE
echo $SLURM_JOB_ID
echo $SLURM_JOB_NODELIST
hostname

if [ "$2" = 'text' ]
then
  PARAM="-D TXT"
else
  PARAM=""
fi

module load intel/19.1.3.304
module load openmpi3/3.1.4
mpicc jacobi_parallel.c -O3 -o jacobi_parallel.out $PARAM


AVE_TEVOLVE=0
AVE_TDATA=0
REPEAT=10
REPETITIONS=$(seq $REPEAT)

for R in $REPETITIONS
do
  OUTPUT=$(mpirun -np $P ./jacobi_parallel.out $DIM $ITER)
  TEVOLVE=${OUTPUT% *}
  TDATA=${OUTPUT#* }
  AVE_TEVOLVE=$(echo "$TEVOLVE+$AVE_TEVOLVE" | bc)
  AVE_TDATA=$(echo "$TDATA+$AVE_TDATA" | bc)
done
AVE_TEVOLVE="$(echo "scale=8; $AVE_TEVOLVE/$REPEAT" | bc)"
AVE_TDATA="$(echo "scale=8; $AVE_TDATA/$REPEAT" | bc)"
F_SAVE="$(echo "scale=8; $AVE_TDATA/$AVE_TEVOLVE/$RATIO" | bc)"
F_SAVE=$(/usr/bin/printf "%.0f" $F_SAVE)
echo "t_evolve = $AVE_TEVOLVE" >> $FILE
echo "t_data = $AVE_TDATA" >> $FILE
echo "f_save = $F_SAVE" >> $FILE

date


