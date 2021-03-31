#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void print_matrix(int m, int n_loc, double* mat, FILE* fp, int save_to_file)
{
  if(save_to_file) {
    fwrite(mat, sizeof(double), m*n_loc, fp);
  } else {
    for(int i = 0; i < n_loc; i++) {
      for(int j = 0; j < m; j++)
	      printf("%.0lf ", mat[i*m+j]);
      printf("\n");
    }
  }
}


void swap(double** a, double** b)
{
  double* tmp = *a;
  *a = *b;
  *b = tmp;
}


int main(int argc, char *argv[])
{
  int rank, p;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  
  int niter = atoi(argv[1]);     // number of iterations
  int n;   // number of rows in the domain
  int m;   // number of columns in the domain

  // the dimensions of the domain are at the beginning of the file
  FILE* fp;
  if(rank==0) {
    fp = fopen("initial_config.dat", "r");
    fread(&n, sizeof(int), 1, fp);
    fread(&m, sizeof(int), 1, fp);
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
  int n_loc = n / p;
  int rest = n % p;
  if(rank < rest) n_loc++;
    
  // the buffer has one additional row for the boundary condition
  double* buffer = (double*)malloc((n_loc+1)*m*sizeof(double));
  double* tmp = (double*)malloc((n_loc+1)*m*sizeof(double));

  // distribute the initial configuration of the domain to all processes
  if(rank==0) {
    fread(buffer, sizeof(double), m*n_loc, fp);
    for(int i=1; i<p; i++) {
      if(i == rest) n_loc -= 1;
      fread(tmp, sizeof(double), m*n_loc, fp);
      MPI_Send(tmp, m*n_loc, MPI_DOUBLE, i, 100, MPI_COMM_WORLD);
    }
    if(rest>0) n_loc += 1;
    fclose(fp);
  } else {
    MPI_Recv(buffer, m*n_loc, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }


  // define source and destination of ring communication
  int dest = (rank - 1 + p)%p;
  int source = (rank + 1)%p;
  MPI_Request request1, request2;
  
  
  // start iterating
	for(int iter=0; iter<niter; iter++) {
	  // post send & receive
	  MPI_Isend(buffer, m, MPI_DOUBLE, dest, 10, MPI_COMM_WORLD, &request1);
	  MPI_Irecv(buffer+n_loc*m, m, MPI_DOUBLE, source, 10, MPI_COMM_WORLD, &request2);
	  
	  // update bulk
	  for(int i=0; i<n_loc-1; i++)
	    for(int j=0; j<m; j++)
	      tmp[i*m+j] = buffer[(i+1)*m+j];
	    
	  // update extremes
	  MPI_Wait(&request2, MPI_STATUS_IGNORE);
	  int i=n_loc-1;
	  for(int j=0; j<m; j++)
	    tmp[i*m+j] = buffer[(i+1)*m+j];
	  
	  // swap buffers
	  MPI_Wait(&request1, MPI_STATUS_IGNORE);
	  swap(&buffer, &tmp);
	}
	
	// collect the final result into one processor and print
  if(rank==0) {
    int save_to_file;
    if(n>=32 || m>=16) save_to_file=1; else save_to_file=0;
    if(save_to_file) {       // same format as the initial_config.dat file
      fp = fopen("final_config.dat", "w");
      fwrite(&n, sizeof(int), 1, fp);
      fwrite(&m, sizeof(int), 1, fp);
    }
    print_matrix(m, n_loc, buffer, fp, save_to_file);
    for(int i=1; i<p; i++) {
      if(i == rest) n_loc -= 1;
      MPI_Recv(buffer, m*n_loc, MPI_DOUBLE, i, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      print_matrix(m, n_loc, buffer, fp, save_to_file);
    }
    if(save_to_file) fclose(fp);
  } else {
    MPI_Send(buffer, m*n_loc, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
  }

  free(buffer);
  free(tmp);
  MPI_Finalize();
  return 0;
}
