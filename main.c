#include "/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define ROOT 0

int malloc_2d_double(double ***array, int n, int m) {

  /* allocate the n*m contiguous items */
  double *p = (double *)malloc(n * m * sizeof(double));
  if (!p)
    return -1;

  /* allocate the row pointers into the memory */
  (*array) = (double **)malloc(n * sizeof(double *));
  if (!(*array)) {
    free(p);
    return -1;
  }

  /* set up the pointers into the contiguous memory */
  for (int i = 0; i < n; i++)
    (*array)[i] = &(p[i * m]);

  return 0;
}

int free_2d_double(double ***array) {
  /* free the memory - the first element of the array is at the start */
  free(&((*array)[0][0]));

  /* free the pointers into the memory */
  free(*array);

  return 0;
}

void inicializar_matriz(double **matriz, int tamano) {
  for (int i = 0; i < tamano; i++) {
    for (int j = 0; j < tamano; j++) {
      matriz[i][j] = (i >= j) ? (double)(rand() % 10 + 1) : 0.0;
    }
  }
}

void imprimir_matriz(double **matriz, int filas, int columnas) {
  for (int i = 0; i < filas; i++) {
    for (int j = 0; j < columnas; j++) {
      printf("%8.3f ", matriz[i][j]);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  int rank, size, N, C;
  double **A = NULL;
  double **local_A = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 3) {
    if (rank == ROOT)
      printf("Uso: %s <N> <C>\n", argv[0]);
    MPI_Finalize();
    return -1;
  }

  // Tamano de la matriz cuadrada
  N = atoi(argv[1]);
  // Tamano de los bloques de columnas
  C = atoi(argv[2]);

  // Solo el proceso 0 inicializa la matriz completa
  if (rank == 0) {
    malloc_2d_double(&A, N, N);
    inicializar_matriz(A, N);
    imprimir_matriz(A, N, N);
  }

  malloc_2d_double(&local_A, N, C);

  /* create a datatype to describe the subarrays of the global array */

  int sizes[2] = {N, N};    /* global size */
  int subsizes[2] = {N, C}; /* local size */
  int starts[2] = {0, 0};   /* where this one starts */
  MPI_Datatype type, resizedtype;
  MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE,
                           &type);
  MPI_Type_create_resized(type, 0, C * sizeof(double), &resizedtype);
  MPI_Type_commit(&resizedtype);

  // Configurar `counts` y `displs` para MPI_Scatterv
  int *counts = malloc(size * sizeof(int));
  int *displs = malloc(size * sizeof(int));

  if (rank == 0) {
    for (int i = 0; i < size; i++) {
      counts[i] = 1; // Cada proceso recibe un bloque
      displs[i] = i;
    }
  }

  MPI_Scatterv((rank == ROOT ? &(A[0][0]) : NULL), counts, displs, resizedtype,
               &(local_A[0][0]), N * C, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

  for (int i = 0; i < size; i++) {
    if (rank == i) {
      printf("Submatriz local en el proceso %d:\n", rank);
      imprimir_matriz(local_A, N, C);
      printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (rank == 0)
    free_2d_double(&A);

  free_2d_double(&local_A);
  free(counts);
  free(displs);

  MPI_Type_free(&resizedtype);
  MPI_Type_free(&type);

  MPI_Finalize();
  return 0;
}
