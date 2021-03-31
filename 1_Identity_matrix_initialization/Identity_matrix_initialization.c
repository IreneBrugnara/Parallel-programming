#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  int rank, p, n, nloc;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  
  if(rank==0) {
    printf("enter dimension of matrix: ");
    fflush(stdout);
  	scanf("%d", &n);
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if(n%p != 0) {
    if(rank==0)
    	printf("error: dimension of matrix not divisible by number of processors\n");
    MPI_Finalize();
    return 0;
  }

  nloc = n/p;
  int* Aloc = (int*)calloc(nloc*n, sizeof(int));
  

  for(int i = rank*nloc; i < n*nloc; i += n+1)
    Aloc[i] = 1;
  


  int* A;
  if(rank==0)
    A = (int*)malloc(n*n*sizeof(int));
    
  MPI_Gather(Aloc, n*nloc, MPI_INT, A, n*nloc, MPI_INT, 0, MPI_COMM_WORLD);
  
  if(rank==0) {
    for(int i=0; i<n; i++) {
      for(int j=0; j<n; j++)
        printf("%d ", A[i*n+j]);  
      printf("\n");
    }
  }
    
  
  MPI_Finalize();
  return 0;
}
