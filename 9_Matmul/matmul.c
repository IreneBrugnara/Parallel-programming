#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <unistd.h>

// serial matrix multiplication
void serial_matmul(double* A, double* B, double* C, int n, int m, int l) {
  for(int i = 0; i < n; i++)
    for(int j = 0; j < l; j++)
      for(int k = 0; k < m; k++)
        C[i*n+j] += A[i*n+k] * B[k*n+j];       // C is supposed to be initialized to zero
}


int main(int argc, char* argv[]) {

  // number of processes
  int rank, p;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  
  // size of matrices
  int n;
  n = atoi(argv[1]);

  // local dimension
  int n_loc;
  int rest, offset;
  n_loc = n / p;
  rest = n % p;
  offset = 0;
  if(rank < rest)
    n_loc++;
  else
    offset += rest;
  int mydispl = rank*n_loc + offset;
    
  int n_loc_max = n/p + (rest>0 ? 1 : 0);   // this is the maximum value that n_loc can take, i.e. n_loc of rank 0
  
  int counts[p];
  int displs[p];
  for(int c=0; c<p; c++) {
    counts[c] = n/p;
    if(c<rest)
      counts[c]++;
    counts[c] *= n_loc_max;
      
    if(c>0)
      displs[c] = counts[c-1] + displs[c-1];
    else
      displs[c] = 0;
  }
  
  
  // buffers
  double *A, *B, *C;
  A = (double*)malloc(n*n_loc*sizeof(double));
  B = (double*)malloc(n*n_loc*sizeof(double));
  C = (double*)calloc(n*n_loc, sizeof(double));
  double* Bloc;   // to gather the columns of B
  Bloc = (double*)malloc(n_loc_max*n*sizeof(double));
  
  
  // initialize A and B with random numbers
  srand(time(NULL) + rank);
  for(int i=0; i<n_loc; i++)
    for(int j=0; j<n; j++) {
      A[i*n+j] = rand() % 10;
      B[i*n+j] = rand() % 10;
    }
  
  
  int current_count, current_displ;
  // main loop
  for(int c=0; c<p; c++) {
    current_count = counts[c]/n_loc_max;
    current_displ = displs[c]/n_loc_max;
  
    // copy my local portion of B to Bloc, to handle non-contiguous data
    for(int i=0; i<n_loc; i++)
      for(int j=0; j<current_count; j++)
        Bloc[ (i+mydispl) * n_loc_max + j ] = B[ i * n + j+current_displ ];
        
    // gather   
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   Bloc, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    // multiply
    for(int i = 0; i < n_loc; i++)
      for(int j = 0; j < current_count; j++)
        for(int k = 0; k < n; k++)
          C[i*n+j+current_displ] += A[i*n+k] * Bloc[k*n_loc_max+j];    // C is supposed to be initialized to zero
    
  }
  


#ifdef TEST
// check if the parallel code works correctly by comparing the result in C with the result of a serial calculation by process 0
  for(int c=0; c<p; c++) {
    counts[c] /= n_loc_max;
    displs[c] /= n_loc_max;
    counts[c] *= n;
    displs[c] *= n;
  }
  
  double *A_all, *B_all, *C_all, *C_correct;
  if(rank==0) {
    A_all = (double*)malloc(n*n*sizeof(double));
    B_all = (double*)malloc(n*n*sizeof(double));
    C_all = (double*)malloc(n*n*sizeof(double));
    C_correct = (double*)calloc(n*n, sizeof(double));
  }
  
  MPI_Gatherv(A, n*n_loc, MPI_DOUBLE,
              A_all, counts, displs, MPI_DOUBLE,
              0, MPI_COMM_WORLD);
              
  MPI_Gatherv(B, n*n_loc, MPI_DOUBLE,
              B_all, counts, displs, MPI_DOUBLE,
              0, MPI_COMM_WORLD);
              
  MPI_Gatherv(C, n*n_loc, MPI_DOUBLE,
              C_all, counts, displs, MPI_DOUBLE,
              0, MPI_COMM_WORLD);
  
  if(rank==0) {
    serial_matmul(A_all, B_all, C_correct, n, n, n);
    
    for(int i=0; i<n; i++)
      for(int j=0; j<n; j++)
        if( C_all[i*n+j] != C_correct[i*n+j] )
          printf("ERROR\n");
          
    free(A_all);
    free(B_all);
    free(C_all);
    free(C_correct);
  }
#endif


  free(A);
  free(B);
  free(C);
  free(Bloc);
  
  MPI_Finalize();

  return 0;
}
