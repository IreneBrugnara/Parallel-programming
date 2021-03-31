#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void swap(int** a, int** b)
{
  int* tmp = *a;
  *a = *b;
  *b = tmp;
}

int main(int argc, char *argv[])
{
  int rank, p;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  
  int n = atoi(argv[1]);
  int* send_buffer = (int*)malloc(n*sizeof(int));
  int* recv_buffer = (int*)malloc(n*sizeof(int));
  for(int i=0; i<n; i++) send_buffer[i] = rank;
  int* sum = (int*)calloc(n, sizeof(int));
  int source = (rank - 1 + p)%p;
  int dest = (rank + 1)%p;
  MPI_Request request1, request2;
  
	for(int iter=0; iter<p; iter++) {
	  MPI_Isend(send_buffer, n, MPI_INT, dest, 10, MPI_COMM_WORLD, &request1);
	  MPI_Irecv(recv_buffer, n, MPI_INT, source, 10, MPI_COMM_WORLD, &request2);
	  for(int i=0; i<n; i++) sum[i] += send_buffer[i];
	  MPI_Wait(&request1, MPI_STATUS_IGNORE);
	  MPI_Wait(&request2, MPI_STATUS_IGNORE);
	  swap(&send_buffer, &recv_buffer);
	}
	

  printf("I am processor %d and my sum[0] is %d\n", rank, sum[0]);
  
  MPI_Finalize();
  return 0;
}
