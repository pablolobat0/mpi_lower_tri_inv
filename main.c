#include "/usr/include/mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define ROOT 0

int malloc_2d_double(double ***array, int n, int m) {
  // allocate the n*m contiguous items
  double *p = (double *)malloc(n * m * sizeof(double));
  if (!p)
    return -1;

  // allocate the row pointers into the memory
  (*array) = (double **)malloc(n * sizeof(double *));
  if (!(*array)) {
    free(p);
    return -1;
  }

  // set up the pointers into the contiguous memory
  for (int i = 0; i < n; i++)
    (*array)[i] = &(p[i * m]);

  return 0;
}

int free_2d_double(double ***array) {
  // free the memory - the first element of the array is at the start
  free(&((*array)[0][0]));

  // free the pointers into the memory
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
      printf("%8.5f ", matriz[i][j]);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  int rank, size, N, C;
  double **A = NULL;
  double **local_A = NULL;
  double **local_A_inv = NULL;

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

  // Inicializacion de la matriz completa
  if (rank == ROOT) {
    if (malloc_2d_double(&A, N, N) == -1) {
      exit(EXIT_FAILURE);
    }
    inicializar_matriz(A, N);
    printf("Matriz inicial:\n");
    imprimir_matriz(A, N, N);
  }

  if (malloc_2d_double(&local_A, N, C) == -1) {
    exit(EXIT_FAILURE);
  }

  if (malloc_2d_double(&local_A_inv, N, C) == -1) {
    exit(EXIT_FAILURE);
  }

  // Inicilizar la matriz inversa local a 0
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < C; j++) {
      local_A_inv[i][j] = 0.0;
    }
  }

  // Creacion de un tipo de dato para describir las submatrices de la matriz
  // global Tamano de la matriz
  int sizes[2] = {N, N};
  // Tamano de la submatriz
  int subsizes[2] = {N, C};
  // Donde empiezan las submatrices
  int starts[2] = {0, 0};
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
      // Cada proceso recibe un bloque
      counts[i] = 1;
      // Saltos entre bloques
      displs[i] = i;
    }
  }

  MPI_Scatterv((rank == ROOT ? &(A[0][0]) : NULL), counts, displs, resizedtype,
               &(local_A[0][0]), N * C, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

  // Se imprimen las submatrices
  for (int i = 0; i < size; i++) {
    if (rank == i) {
      printf("Submatriz local en el proceso %d:\n", rank);
      imprimir_matriz(local_A, N, C);
      printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  for (int i = 0; i < N; i++) {
  }

  double *fila = (double *)malloc(N * sizeof(double));

  // Calcular los elementos por debajo de la diagonal
  for (int i = 0; i < N; i++) {
    // Calcular el elemento de la diagonal
    int diagonal_owner = i / C;
    if (rank == diagonal_owner) {
      int local_col = i % C;
      local_A_inv[i][local_col] = 1.0 / local_A[i][local_col];
    }

    // Root rellena la fila en la que van a trabajar los procesos
    if (rank == ROOT) {
      for (int a = 0; a < N; a++) {
        fila[a] = A[i][a];
      }
    }

    MPI_Bcast(fila, N, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    // Calculo de la inversa para cada elemento de la fila
    for (int j = 0; j < C; j++) {
      if (i > (j + C * rank)) {
        for (int k = 0; k < i; k++) {
          local_A_inv[i][j] += fila[k] * local_A_inv[k][j];
        }
        local_A_inv[i][j] /= -fila[i];
      }
    }
  }

  // Se devuelven las submatrices inversas calculadas al proceso root
  MPI_Gatherv(&(local_A_inv[0][0]), C * N, MPI_DOUBLE,
              (rank == ROOT ? &(A[0][0]) : NULL), counts, displs, resizedtype,
              ROOT, MPI_COMM_WORLD);

  // Impresion de la matriz inversa
  if (rank == ROOT) {
    printf("Matriz inversa:\n");
    imprimir_matriz(A, N, N);
    free_2d_double(&A);
  }

  free_2d_double(&local_A);
  free(counts);
  free(displs);
  free(fila);

  MPI_Type_free(&resizedtype);
  MPI_Type_free(&type);

  MPI_Finalize();
  return 0;
}
