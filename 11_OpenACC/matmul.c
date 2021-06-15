#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <openacc.h>


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
  double time_comm=0;
  
  // size of matrices
  int n;
  if(argc!=2) {
    printf("wrong number of args\n");
    exit(1);
  }
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
      
  
  
  long int dim = n*n_loc;
  long int dim_max = n*n_loc_max;
  long int size = dim*sizeof(double);
  long int size_max = dim_max*sizeof(double);
  
  
  // allocate on host
  double *A, * restrict B, *C;
  A = (double*)malloc(size);
  B = (double*)malloc(size);
  C = (double*)malloc(size);
  double* restrict Bloc;   // to gather the columns of B
  // "restrict" is to tell the compiler that there is no pointer aliasing, otherwise you cannot
  // parallelize the loop below with B and Bloc
  Bloc = (double*)malloc(size_max);  // this allocation on CPU would be unnecessary if I used acc_malloc instead of create(Bloc), but for the sake of readability I prefer to do like this
  
  
  // initialize A and B with random numbers
  start_time = MPI_Wtime();
  #pragma omp parallel
  {
    unsigned int seed = time(NULL) ^ rank ^ omp_get_thread_num();
    #pragma omp for
    for(int i=0; i<n_loc; i++)
      for(int j=0; j<n; j++) {
        A[i*n+j] = rand_r(&seed) % 10;
        B[i*n+j] = rand_r(&seed) % 10;
      }
  }
  end_time = MPI_Wtime();
#ifdef VERBOSE
  if(rank==0) printf("random init = %lf\n", end_time-start_time);
#endif



  // associate MPI process to GPU
  start_time = MPI_Wtime();
  
  int num_devices = acc_get_num_devices(acc_device_nvidia);  // number of GPUs per node (2)
  int id_device = rank % num_devices;   // id of the GPU (0 or 1)
  acc_set_device_num(id_device, acc_device_nvidia);
  
  end_time = MPI_Wtime();
#ifdef VERBOSE
  if(rank==0) printf("setdevice  = %lf\n", end_time-start_time);
#endif
  
  // handle for Cublas (the handle will use the above device)
  start_time = MPI_Wtime();
  
  cublasHandle_t handle;
  cublasStatus_t status;
  status = cublasCreate(&handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "failed\n");
      return EXIT_FAILURE;
    }
  
  
  end_time = MPI_Wtime();
#ifdef VERBOSE
  if(rank==0) printf("handle  = %lf\n", end_time-start_time);
#endif


  
  double alpha = 1;
  double beta = 0;
  int current_count, current_displ;
  
  
  // initialize counts and displacements for allgather
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

  start_time = MPI_Wtime();
  // main loop
  # pragma acc data copyin(A[0:dim]) copyin(B[0:dim]) create(Bloc[0:dim_max]) copyout(C[0:dim])
  for(int c=0; c<p; c++) {

    current_count = counts[c]/n_loc_max;
    current_displ = displs[c]/n_loc_max;

    
    start_sub_time = MPI_Wtime();
  
    // copy my local portion of B to Bloc, to handle non-contiguous data
    # pragma acc declare present(B, Bloc)
    # pragma acc parallel loop
    for(int i=0; i<n_loc; i++)
      for(int j=0; j<current_count; j++)
        Bloc[ (i+mydispl) * n_loc_max + j ] = B[ i * n + j+current_displ ];

    
    // gather   
    # pragma acc host_data use_device(Bloc)
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   Bloc, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
                   
    
    end_sub_time = MPI_Wtime();
    time_comm += end_sub_time - start_sub_time;
    

    // multiply (A and Bloc are exchanged because cublas wants column-major)
    # pragma acc host_data use_device(Bloc, A, C)
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, current_count, n_loc, n,
                         &alpha, Bloc, n_loc_max, A, n, &beta, C+current_displ, n);
                         
    # pragma acc wait
    
  }
  
  end_time = MPI_Wtime();

  

#ifdef VERBOSE
  if(rank==0) printf("time of communication: %lf\n", time_comm);
  if(rank==0) printf("core: %lf\n", end_time-start_time);
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
    printf("allright\n");
  }
#endif


  free(A);
  free(B);
  free(C);
 

  
  
  MPI_Finalize();

  return 0;
}
