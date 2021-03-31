#include <stdio.h>
#include <stdlib.h>

double main(int argc, char *argv[])
{
  int n = atoi(argv[1]);  // number of rows in the domain
  int m=7;                // number of columns in the domain
  double* buffer = (double*)calloc(n*m, sizeof(double));
  
  // draw a "particle" in the domain
  buffer[(n-1)*m+ 2] = 1;
  buffer[(n-1)*m+ 3] = 1;
  buffer[(n-2)*m+ 2] = 1;
  buffer[(n-2)*m+ 3] = 1;
  
  // write the configuration to a file
  FILE* fp = fopen("initial_config.dat", "w");
  // put the dimensions at the beginning of the file
  fwrite(&n, sizeof(int), 1, fp);
  fwrite(&m, sizeof(int), 1, fp);
  fwrite(buffer, sizeof(double), n*m, fp);
  fclose(fp);
}
