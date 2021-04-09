// code adapted from https://github.com/sissa/dssc_jacobi

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <unistd.h>


/*** function declarations ***/

// evolve Jacobi
void evolve_borders( double * matrix, double *matrix_new, size_t dimension, size_t d_loc );
void evolve_bulk( double * matrix, double *matrix_new, size_t dimension, size_t d_loc );

// communication between processes
void exchange_boundaries( double *matrix, size_t d_loc, size_t dimension, int above_rank, int below_rank );
void exchange_boundaries_nonblocking( double *matrix, size_t d_loc, size_t dimension, int above_rank, int below_rank, MPI_Request **request );

// save matrix to file
void print_timestep(double *matrix, double *matrix_new, size_t d_loc, size_t dimension, int rank, int p, int rest, int offset, int iter);
void print_matrix(double* mat, size_t d_loc, size_t dimension, int include_upper_border, int include_lower_border);
void save_data_txt( double *matrix, size_t d_loc, size_t dimension, FILE* fp, int rank, int offset, int include_upper_border, int include_lower_border );
void save_data_binary( double *matrix, size_t d_loc, size_t dimension, FILE* fp, int rank, int p, int include_upper_border, int include_lower_border );

// swap pointers
void swap(double** a, double** b);

/*** end function declaration ***/



int main(int argc, char* argv[]){

  // number of processes
  int rank, p;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // timing variables
  double start_comm, end_comm, start_comp, end_comp;   
  double time_comm=0, time_comp=0; 

  // indexes for loops
  size_t i, j, i_global, it;
  
  // buffers
  double *matrix, *matrix_new;

  size_t dimension = 0, iterations = 0, row_peek = 0, col_peek = 0;
  size_t byte_dimension = 0;
  
  double increment;
  int f_dump;

  // check on input parameters
  if(argc != 4) {
    if(rank==0)
      fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it fdump\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  dimension = atoi(argv[1]);
  iterations = atoi(argv[2]);
  f_dump = atoi(argv[3]);

  
  // local dimension
  size_t d_loc;
  int rest, offset;
  d_loc = dimension / p;
  rest = dimension % p;
  offset = 0;
  if(rank < rest)
    d_loc++;
  else
    offset += rest;


  // initialize matrix
  byte_dimension = sizeof(double) * ( d_loc + 2 ) * ( dimension + 2 );
  matrix = ( double* )malloc( byte_dimension );
  matrix_new = ( double* )malloc( byte_dimension );

  memset( matrix, 0, byte_dimension );
  memset( matrix_new, 0, byte_dimension );

  //fill initial values  
  for( i = 1; i <= d_loc; ++i )
    for( j = 1; j <= dimension; ++j )
      matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;
	      
  // set up borders 
  increment = 100.0 / ( dimension + 1 );
  
  // left border
  for( i=1; i <= d_loc; ++i ){
    i_global = rank*d_loc + offset + i;
    matrix[ i * ( dimension + 2 ) ] = i_global * increment;
    matrix_new[ i * ( dimension + 2 ) ] = i_global * increment;

  }
  
  // bottom border
  if( rank==p-1 ) {
    for( j=1; j <= dimension+1; ++j ){
      matrix[ ( ( d_loc + 1 ) * ( dimension + 2 ) ) + ( dimension + 1 - j ) ] = j * increment;
      matrix_new[ ( ( d_loc + 1 ) * ( dimension + 2 ) ) + ( dimension + 1 - j ) ] = j * increment;
    }
  }
  
  
  // define source and destination of communications
  int above_rank = rank==0 ? MPI_PROC_NULL : (rank - 1);
  int below_rank = rank==p-1 ? MPI_PROC_NULL : (rank + 1);
  MPI_Request* request;
  request = (MPI_Request *) malloc( 4 * sizeof(MPI_Request) );

    
  // start algorithm
  for( it = 0; it < iterations; ++it ){
    start_comm = MPI_Wtime();
#ifdef BLOCKING
    exchange_boundaries( matrix, d_loc, dimension, above_rank, below_rank );
#else
    exchange_boundaries_nonblocking( matrix, d_loc, dimension, above_rank, below_rank, &request );
#endif
    end_comm = MPI_Wtime();
    time_comm += end_comm - start_comm;
    
    start_comp = MPI_Wtime();

    evolve_bulk( matrix, matrix_new, dimension, d_loc );

#ifndef BLOCKING
    end_comp = MPI_Wtime();
    time_comp += end_comp - start_comp;
    start_comm = MPI_Wtime();
    
    MPI_Waitall( 4, request, MPI_STATUS_IGNORE );
    
    end_comm = MPI_Wtime();
    time_comm += end_comm - start_comm;
    start_comp = MPI_Wtime();
#endif

    evolve_borders( matrix, matrix_new, dimension, d_loc );

    swap( &matrix, &matrix_new );
    
    end_comp = MPI_Wtime();
    time_comp += end_comp - start_comp;
    
    
#ifdef PRINT
    if( it % f_dump == 0 ) {
      print_timestep( matrix, matrix_new, d_loc, dimension, rank, p, rest, offset, it );
    }
#endif

  }
  

  if(rank==0) {
    printf("%lf %lf\n", time_comm, time_comp);
  }
  

  
  free( matrix );
  free( matrix_new );
  
  MPI_Finalize();

  return 0;
}

void exchange_boundaries( double *matrix, size_t d_loc, size_t dimension, int above_rank, int below_rank )
{
  size_t augm_dim = dimension+2;
  // send my first row and receive the bottom boundary
  MPI_Sendrecv( matrix + augm_dim, augm_dim, MPI_DOUBLE, above_rank, 100,
                matrix + (d_loc + 1) * augm_dim, augm_dim, MPI_DOUBLE, below_rank, 100,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE );
  // send my last row and receive the top boundary
  MPI_Sendrecv( matrix + (d_loc * augm_dim), augm_dim, MPI_DOUBLE, below_rank, 100,
                matrix, augm_dim, MPI_DOUBLE, above_rank, 100,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE );
                
}

void exchange_boundaries_nonblocking( double *matrix, size_t d_loc, size_t dimension, int above_rank, int below_rank, MPI_Request **request )
{
  size_t augm_dim = dimension+2;
  // send my first row
  MPI_Isend( matrix + augm_dim, augm_dim, MPI_DOUBLE, above_rank, 100, MPI_COMM_WORLD, (* request + 0) );
  // receive the bottom boundary
  MPI_Irecv( matrix + ( (d_loc + 1) * augm_dim), augm_dim, MPI_DOUBLE, below_rank, 100, MPI_COMM_WORLD, (* request + 1) );
  // send my last row
  MPI_Isend( matrix + (d_loc * augm_dim), augm_dim, MPI_DOUBLE, below_rank, 300, MPI_COMM_WORLD, (* request + 2) );
  // receive the top boundary 
  MPI_Irecv( matrix, augm_dim, MPI_DOUBLE, above_rank, 300, MPI_COMM_WORLD, (* request + 3) );
}

void evolve_bulk( double * matrix, double *matrix_new, size_t dimension, size_t d_loc )
{
  size_t i , j;
  for( i = 2 ; i < d_loc; ++i )
    for( j = 1; j <= dimension; ++j )
      matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) * 
	      ( matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] + 
	        matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] + 	  
	        matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] + 
	        matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ] ); 
}

void evolve_borders( double * matrix, double *matrix_new, size_t dimension, size_t d_loc )
{
  size_t i , j;
  for( j = 1; j <= dimension; ++j ) {
    i = 1;
    matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) * 
	    ( matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] + 
	      matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] + 	  
	      matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] + 
	      matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ] ); 
	  i = d_loc;
	  matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) * 
	    ( matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] + 
	      matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] + 	  
	      matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] + 
	      matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ] ); 
	}
}

void swap(double** a, double** b)
{
  double* tmp = *a;
  *a = *b;
  *b = tmp;
}

void print_timestep(double *matrix, double *matrix_new, size_t d_loc, size_t dimension, int rank, int p, int rest, int offset, int iter)
{
  if(rank==0) {
    int include_upper_border=1, include_lower_border=0;
    if(rank==p-1) include_lower_border=1;  // if rank 0 has the last row i.e. p=1, must print it

    char filename[30];
#ifdef TXT
    sprintf(filename, "solution_it%d.txt", iter);
    FILE* fp = fopen( filename, "w" );
    save_data_txt( matrix, d_loc, dimension, fp, rank, p, include_upper_border, include_lower_border );
#else
    sprintf(filename, "solution_it%d.dat", iter);
    FILE* fp = fopen( filename, "wb" );
    save_data_binary( matrix, d_loc, dimension, fp, rank, p, include_upper_border, include_lower_border );
#endif
    include_upper_border=0;
    include_lower_border=0;
    for(int i=1; i<p; i++) {
      if(i == rest) d_loc -= 1;
      if(i == p-1) include_lower_border=1;
      MPI_Recv(matrix_new+(dimension+2), (dimension+2)*(d_loc+include_lower_border), MPI_DOUBLE,
               i, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // I can avoid receiving the borders because I don't need them (except the first and last)

#ifdef TXT
      save_data_txt( matrix_new, d_loc, dimension, fp, i, p, include_upper_border, include_lower_border );
#else
      save_data_binary( matrix_new, d_loc, dimension, fp, i, p, include_upper_border, include_lower_border );
#endif
    }
    fclose( fp );
    // ripristino il bordo sinistro di matrix_new del rank 0 con le condizioni al contorno:
    if(rest < p) d_loc += 1;
    double increment = 100.0 / ( dimension + 1 );
    for(size_t i=1; i <= d_loc; ++i )
      matrix_new[ i * ( dimension + 2 ) ] = i * increment;
  } else {
    int include_upper_border=0, include_lower_border=0;
    if(rank==p-1) include_lower_border=1;  // the last processor needs to send also the last row of matrix which is the boundary
    MPI_Send(matrix+(dimension+2), (dimension+2)*(d_loc+include_lower_border), MPI_DOUBLE,
             0, 200, MPI_COMM_WORLD);
  }
}


void save_data_txt( double *matrix, size_t d_loc, size_t dimension, FILE* fp, int rank, int p, int include_upper_border, int include_lower_border ){

  int rest = dimension % p;
  int offset = 0;
  if( rank >= rest ) offset += rest;
  size_t shift = rank*d_loc + offset;
  
  size_t start_i = 1;
  if(include_upper_border) start_i -= 1;
  if(include_lower_border) d_loc += 1;
  
  size_t i , j;
  const double h = 0.1;
  for( i = start_i; i <= d_loc ; ++i ) {
    for( j = 0; j < dimension+2; ++j ) {
      fprintf(fp, "%.2f ", matrix[ i * (dimension+2) + j ] );
    }
    fprintf(fp, "\n");
  }
}

void save_data_binary( double *matrix, size_t d_loc, size_t dimension, FILE* fp, int rank, int p, int include_upper_border, int include_lower_border ){

  int rest = dimension % p;
  int offset = 0;
  if( rank >= rest ) offset += rest;
  size_t shift = rank*d_loc + offset;
  
  size_t start_i = 1;
  if(include_upper_border) start_i -= 1;
  if(include_lower_border) d_loc += 1;
  
  const double h = 0.1;
  fwrite(matrix+start_i*(dimension+2), sizeof(double), (dimension+2)*(d_loc-start_i+1), fp);
}

void print_matrix(double* mat, size_t d_loc, size_t dimension, int include_upper_border, int include_lower_border)
{
  size_t start_i = 1;
  if(include_upper_border) start_i -= 1;
  if(include_lower_border) d_loc += 1;
  size_t i , j;
  
  for( i=start_i; i <= d_loc; i++) {
    for( j = 0; j < dimension+2; j++)
	    printf("%.2lf\t", mat[i*(dimension+2)+j]);
    printf("\n");
  }
}


