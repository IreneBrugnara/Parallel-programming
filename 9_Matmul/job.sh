#!/bin/bash
#SBATCH --job-name=matmul
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:50:00
#SBATCH --partition=regular2
#SBATCH --output=log.txt
#SBATCH --hint=nomultithread
#SBATCH --mem=40G

date

#if [ "$1" = 'big' ]
#then
#  DIM=32000
#else    # $2 = 'small'
#  DIM=6400
#fi
DIM=$1

NODES="1 2"


FILE="time_"$1"_"$2"_"$3".txt"
> $FILE
echo $SLURM_JOB_ID
echo $SLURM_JOB_NODELIST
hostname

module load intel/19.1.3.304
module load openmpi3/3.1.4


if [ "$2" = 'mkl' ]
then
  DGEMM_FLAG="-D__DGEMM"
  COMPILE_FLAG="-I/opt/sissa/compiler/intel/2020.4/compilers_and_libraries_2020.4.304/linux/mkl/include"
  LINK_FLAG="-L/opt/sissa/compiler/intel/2020.4/compilers_and_libraries_2020.4.304/linux/mkl/lib -lmkl_intel_lp64 -lmkl_core"
  if [ "$3" = 'thread' ]
  then
    THREAD_FLAG="-lmkl_intel_thread"
  else
    THREAD_FLAG="-lmkl_sequential"
  fi
fi


# compile
mpicc -O3 -c matmul.c $DGEMM_FLAG $COMPILE_FLAG -qopenmp


# link
mpicc -O3 -o matmul.x matmul.o $DGEMM_FLAG $LINK_FLAG $THREAD_FLAG -qopenmp



for N in $NODES
do
  echo "running N=$N"
  if [ "$3" = 'thread' ]    # run the hybrid MPI + openMP version
  then
    P=$((2*$N))
    export OMP_NUM_THREADS=16
    OUTPUT=$(mpirun -np $P --map-by ppr:1:socket:pe=16 ./matmul.x $DIM)
  else # serial   # run the MPI only version
    P=$((32*$N))
    export OMP_NUM_THREADS=1
    OUTPUT=$(mpirun -np $P --map-by ppr:16:socket:pe=1 ./matmul.x $DIM)
  fi
  TCOMM=${OUTPUT% *}
  TCOMP=${OUTPUT#* }
  echo "$N $TCOMM $TCOMP" >> $FILE
done


date


