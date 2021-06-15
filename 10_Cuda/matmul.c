#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>

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
  
  double start_time, end_time, start_sub_time, end_sub_time;
  double time_comm=0, time_comp=0, time_copy=0; 
  
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

    
  
  long int size = n*n_loc*sizeof(double);
  long int size_max = n_loc_max*n*sizeof(double);
  
  // allocate on host
  double *h_A, *h_B, *h_C;
  h_A = (double*)malloc(size);
  h_B = (double*)malloc(size);
  h_C = (double*)malloc(size);

  double* h_Bloc;   // to gather the columns of B
  h_Bloc = (double*)malloc(size_max);
  
  
  // initialize A and B with random numbers
  start_time = MPI_Wtime();
  #pragma omp parallel
  {
    unsigned int seed = time(NULL) ^ rank ^ omp_get_thread_num() + rank;
    #pragma omp for
    for(int i=0; i<n_loc; i++)
      for(int j=0; j<n; j++) {
        h_A[i*n+j] = rand_r(&seed) % 10;
        h_B[i*n+j] = rand_r(&seed) % 10;
      }
  }
  end_time = MPI_Wtime();
#ifdef VERBOSE
  if(rank==0) printf("random init = %lf\n", end_time-start_time);
#endif


  // associate MPI process to GPU
  int num_devices=0;
  cudaGetDeviceCount(&num_devices);
    
  start_time = MPI_Wtime();

  cudaSetDevice(rank%num_devices);   // 2 is the number of GPUs per node, and we associate 1 process to each GPU
  
  end_time = MPI_Wtime();
#ifdef VERBOSE
  if(rank==0) printf("setdevice  = %lf\n", end_time-start_time);
#endif
  
  // handle for Cublas (the handle will use the above device)
  start_time = MPI_Wtime();
  cublasHandle_t handle;
  cublasStatus_t status;
  status = cublasCreate(&handle);
  end_time = MPI_Wtime();
#ifdef VERBOSE
  if(rank==0) printf("handle  = %lf\n", end_time-start_time);
#endif

  start_time = MPI_Wtime();
  
  // allocate on device
  cudaError_t err = cudaSuccess;
  double *d_A, *d_Bloc, *d_C;
  
  err = cudaMalloc((void**)&d_A, size);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device matrix A (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void**)&d_C, size);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device matrix C (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void**)&d_Bloc, size_max);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device matrix Bloc (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  
  

  
  end_time = MPI_Wtime();
#ifdef VERBOSE
  if(rank==0) printf("alloc on gpu = %lf\n", end_time-start_time);
#endif
  

  start_time = MPI_Wtime();
    
  // transfer A to GPU
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  end_sub_time = MPI_Wtime();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy A to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
#ifdef VERBOSE
  if(rank==0) printf("time to transfer A: %lf\n", end_sub_time-start_time);
#endif
  
  double alpha = 1;
  double beta = 0;
  int current_count, current_displ;
  
  
  for(int c=0; c<p; c++) {
    current_count = counts[c]/n_loc_max;
    current_displ = displs[c]/n_loc_max;
    
    start_sub_time = MPI_Wtime();
  
    // copy my local portion of B to Bloc, to handle non-contiguous data
    # pragma omp parallel for
    for(int i=0; i<n_loc; i++)
      for(int j=0; j<current_count; j++)
        h_Bloc[ (i+mydispl) * n_loc_max + j ] = h_B[ i * n + j+current_displ ];
        
        
    // gather   
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   h_Bloc, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
                   
    end_sub_time = MPI_Wtime();
    time_comm += end_sub_time - start_sub_time;
    
    // transfer Bloc to GPU
    err = cudaMemcpy(d_Bloc, h_Bloc, size_max, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to copy Bloc to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    

    // multiply
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, current_count, n_loc, n,
                         &alpha, d_Bloc, n_loc_max, d_A, n, &beta, d_C+current_displ, n);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "!!!! kernel execution error.\n");
      return EXIT_FAILURE;
    }
  
  }
  


  
  // transfer C back
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy C to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  end_time = MPI_Wtime();
  
#ifdef VERBOSE
  if(rank==0) printf("time of communication: %lf\n", time_comm);
  if (rank==0) printf("core: %lf\n", end_time-start_time);
#else
   if(rank==0) printf("%lf %lf\n", end_time-start_time, time_comm);
#endif
  
 
  


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
  
  MPI_Gatherv(h_A, n*n_loc, MPI_DOUBLE,
              A_all, counts, displs, MPI_DOUBLE,
              0, MPI_COMM_WORLD);
              
  MPI_Gatherv(h_B, n*n_loc, MPI_DOUBLE,
              B_all, counts, displs, MPI_DOUBLE,
              0, MPI_COMM_WORLD);
              
  MPI_Gatherv(h_C, n*n_loc, MPI_DOUBLE,
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
    printf("allright\n");
  }
#endif


  free(h_A);
  free(h_B);
  free(h_C);
  free(h_Bloc);
  
  cudaFree(d_A);
  cudaFree(d_Bloc);
  cudaFree(d_C);
  
  
  MPI_Finalize();

  return 0;
}
