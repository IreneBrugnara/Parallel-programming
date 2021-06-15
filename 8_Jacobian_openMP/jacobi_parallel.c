#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <hdf5.h>
#include <omp.h>

/*** function declarations ***/

// evolve Jacobi
void evolve_borders( double * matrix, double *matrix_new, size_t dimension, size_t d_loc );
void evolve_bulk( double * matrix, double *matrix_new, size_t dimension, size_t d_loc );

// communication between processes
void exchange_boundaries( double *matrix, size_t d_loc, size_t dimension, int above_rank, int below_rank );
void exchange_boundaries_nonblocking( double *matrix, size_t d_loc, size_t dimension, int above_rank, int below_rank, MPI_Request **request );

// save matrix to file
void print_hdf5_time_evolution( double * matrix, size_t d_loc, size_t dimension, int rank, int npes, int rest, int it );
void print_hdf5_time_evolution_par( double * matrix, size_t d_loc, size_t dimension, int rank, int npes, int rest, int offset, int it );

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
  double t_comp=0, t_comm=0, t_data=0, increment;
  double start_time, end_time;   
  int f_dump;

  // indexes for loops
  size_t i, j, i_global, it;
  
  // buffers
  double *matrix, *matrix_new;

  size_t dimension = 0, iterations = 0, row_peek = 0, col_peek = 0;
  size_t byte_dimension = 0;

  // check on input parameters
  if(argc != 4) {
    if(rank==0)
      fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it\n");
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
    start_time = MPI_Wtime();
#ifdef BLOCKING
    exchange_boundaries( matrix, d_loc, dimension, above_rank, below_rank );
#else
    exchange_boundaries_nonblocking( matrix, d_loc, dimension, above_rank, below_rank, &request );
#endif
    end_time = MPI_Wtime();
    t_comm += end_time - start_time;
    
    start_time = MPI_Wtime();

    evolve_bulk( matrix, matrix_new, dimension, d_loc );

#ifndef BLOCKING
    end_time = MPI_Wtime();
    t_comp += end_time - start_time;
    start_time = MPI_Wtime();
    
    MPI_Waitall( 4, request, MPI_STATUS_IGNORE );
    
    end_time = MPI_Wtime();
    t_comm += end_time - start_time;
    start_time = MPI_Wtime();
#endif

    evolve_borders( matrix, matrix_new, dimension, d_loc );

    swap( &matrix, &matrix_new );
    
    end_time = MPI_Wtime();
    t_comp += end_time - start_time;

    
    if(it % 10 == 0) {
      start_time = MPI_Wtime();
#ifndef PAR
      print_hdf5_time_evolution( matrix, d_loc, dimension, rank, p, rest, iterations );
#else
      print_hdf5_time_evolution_par( matrix, d_loc, dimension, rank, p, rest, offset, iterations );
#endif
      end_time = MPI_Wtime();
      t_data += end_time - start_time;
    }

  }
  
  
  if(rank==0) {
    printf("%f\n%f\n%f\n", t_comp, t_comm, t_data);
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
  MPI_Isend( matrix + (d_loc * augm_dim), augm_dim, MPI_DOUBLE, below_rank, 100, MPI_COMM_WORLD, (* request + 2) );
  // receive the top boundary 
  MPI_Irecv( matrix, augm_dim, MPI_DOUBLE, above_rank, 100, MPI_COMM_WORLD, (* request + 3) );
}

void evolve_bulk( double * matrix, double *matrix_new, size_t dimension, size_t d_loc )
{
  size_t i , j;
  #pragma omp parallel for
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
  #pragma omp parallel for
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

void print_hdf5_time_evolution_par( double * matrix, // memeory buffer 
				    size_t d_loc, // local dimension (boundaries excluded)
				    size_t dimension, // global dimension (boundaries excluded)
				    int rank, 
				    int npes, 
				    int rest, // rest of the data assigned to a given process
				    int offset, // offset regarding the rest distribution assigned to a given process     
				    int it )  // iteration   
{
  // Init Par HDF5
  H5Eset_current_stack (H5E_DEFAULT);
  hid_t plist_id = H5Pcreate (H5P_FILE_ACCESS);
  hid_t hdf5_status = H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  
  hid_t   file_id, group_id,dataset_id,dataspace_id;  /* identifiers */
  herr_t  status;
  
  char fname[100];
  sprintf(fname,"test_%d.h5", it);
  /* Open an existing HDF5 file or create if not existing. */
  file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    
  /* set the file dimensions = data domain dimension, in our case */
  hsize_t dims_data[ 2 ] = { dimension + 2, dimension + 2 };
  hid_t file_space = H5Screate_simple(2, dims_data, NULL);
  dataset_id = H5Dcreate( file_id, "/temp", H5T_NATIVE_DOUBLE, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
  
  /* set the dimensions of the local data dumped by process 0 */
  dims_data[ 0 ] = !rank || rank == npes - 1 ? d_loc + 1 : d_loc;
  if( npes == 1 ) dims_data[ 0 ] += 1; 
  dims_data[ 1 ] = dimension + 2;

  hid_t mem_space = H5Screate_simple(2, dims_data, NULL);
  
  /* set the global starting point of the 0 process slice in regards to the file dimensions (global domain dimensions) */ 
  hsize_t start_3d_mem[ 2 ];
  start_3d_mem[ 0 ] = rank ? d_loc * rank + offset + 1 : 0;
  start_3d_mem[ 1 ] = 0;
  
  /* set the hyperslab (portion) of the data to write on the file */
  status = H5Sselect_hyperslab ( file_space, H5S_SELECT_SET, start_3d_mem, NULL, dims_data, NULL );
  
    /* set pbject for par IO */
  hid_t xfer_plist = H5Pcreate (H5P_DATASET_XFER);
  herr_t ret = H5Pset_dxpl_mpio (xfer_plist, H5FD_MPIO_COLLECTIVE);
  
  /* write the portion local to process 0 */
  status = H5Dwrite( dataset_id, H5T_NATIVE_DOUBLE, mem_space, file_space, xfer_plist, rank ? matrix + (dimension+2) : matrix );
  
  // relase the objects to avoid memory leaks
  status = H5Sclose( mem_space );
  status = H5Pclose( xfer_plist );
  status = H5Pclose( plist_id );
  status = H5Sclose( file_space );
  status = H5Dclose( dataset_id );    
  status = H5Fclose(file_id);
}


void print_hdf5_time_evolution( double * matrix, // memeory buffer 
				size_t d_loc, // local dimension (boundaries excluded)
				size_t dimension, // global dimension (boundaries excluded)
				int rank, 
				int npes, 
				int rest, // rest of the data assigned to a given process
				int it )  // iteration   
{
  int include_lower_border = 0;
  int offset = 0;

  if( !rank ) {

    if( rank == npes - 1 ) include_lower_border = 1;  // if rank 0 has the last row i.e. npes =1, must print it
    
    hid_t   file_id, group_id,dataset_id,dataspace_id;  /* identifiers */
    herr_t  status;
    
    char fname[100];
    sprintf(fname,"test_%d.h5", it);
    /* Open an existing HDF5 file or create if not existing. */
    file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    /* set the file dimensions = data domain dimension, in our case */
    hsize_t dims_data[ 2 ] = { dimension + 2, dimension + 2 };
    hid_t file_space = H5Screate_simple(2, dims_data, NULL);
    dataset_id = H5Dcreate( file_id, "/temp", H5T_NATIVE_DOUBLE, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

    /* set the dimensions of the local data dumped by process 0 */
    dims_data[ 0 ] = d_loc + include_lower_border + 1;
    dims_data[ 1 ] = dimension + 2;
    hid_t mem_space = H5Screate_simple(2, dims_data, NULL);

    /* set the global starting point of the 0 process slice in regards to the file dimensions (global domain dimensions) */ 
    hsize_t start_3d_mem[ 2 ];
    start_3d_mem[ 0 ] = 0;
    start_3d_mem[ 1 ] = 0;

    /* set the hyperslab (portion) of the data to write on the file */
    status = H5Sselect_hyperslab ( file_space, H5S_SELECT_SET, start_3d_mem, NULL, dims_data, NULL );
    
    /* write the portion local to process 0 */
    status = H5Dwrite( dataset_id, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, matrix );
    
    int i = 0;
    for( i = 1; i < npes; i++ ){

      if( i == rest ){
	d_loc -= 1;
	offset = rest;
      }

      if( i == npes - 1 ) include_lower_border = 1;

      MPI_Recv( matrix, ( dimension + 2 ) * ( d_loc + include_lower_border ), MPI_DOUBLE, i, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      /* set the dimensions of the data to print as received from process i */
      H5Sclose( mem_space );
      dims_data[ 0 ] = d_loc + include_lower_border;
      dims_data[ 1 ] = dimension + 2;
      mem_space = H5Screate_simple( 2, dims_data, NULL );

      /* set the global starting point of the ith process slice in regards to the file dimensions (global domain dimensions) */ 
      start_3d_mem[ 0 ] = d_loc * i + 1 + offset;
      start_3d_mem[ 1 ] = 0;

      status = H5Sselect_hyperslab ( file_space, H5S_SELECT_SET, start_3d_mem, NULL, dims_data, NULL );
      status = H5Dwrite( dataset_id, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, matrix );
    }
    // relase the objects to avoid memory leaks
    status = H5Sclose( mem_space );
    status = H5Sclose( file_space );
    status = H5Dclose( dataset_id );    
    status = H5Fclose( file_id );
    
  } else {
    if( rank == npes - 1 ) d_loc++;
    MPI_Send( matrix + ( dimension + 2 ), ( dimension + 2 ) * d_loc, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
  }
}


