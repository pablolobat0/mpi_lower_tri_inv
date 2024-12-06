#include "/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"
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

  // Reservar memoria dinámica para las submatrices locales
  if (malloc_2d_double(&local_A, N, N / size) == -1) {
    exit(EXIT_FAILURE);
  }

  if (malloc_2d_double(&local_A_inv, N, N / size) == -1) {
    exit(EXIT_FAILURE);
  }

  // Inicializacion de la matriz inversa local a 0
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N / size; j++) {
      local_A_inv[i][j] = 0.0;
    }
  }

  // Crear un tipo derivado para un bloque de columnas
  MPI_Datatype block, block_type;
  MPI_Type_vector(N, 1, N, MPI_DOUBLE, &block);
  MPI_Type_commit(&block);
  MPI_Type_create_resized(block, 0, 1 * sizeof(double), &block_type);
  MPI_Type_commit(&block_type);

  int x = 0;
  for (int a = 0; a <= N; a += size) {
    double *recv_buffer = (double *)malloc(N * sizeof(double));
    MPI_Scatter((rank == ROOT ? &(A[0][a]) : NULL), 1, block_type, recv_buffer,
                N, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    for (int j = 0; j < N; j++) {
      local_A[j][x] = recv_buffer[j];
    }
    x++;
  }
  // Se imprimen las submatrices
  for (int i = 0; i < size; i++) {
    if (rank == i) {
      printf("Submatriz local en el proceso %d:\n", rank);
      imprimir_matriz(local_A, N, N / size);
      printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  double *fila = (double *)malloc(N * sizeof(double));

  // Calcular los elementos por debajo de la diagonal
  for (int i = 0; i < N; i++) {
    // Calcular el elemento de la diagonal
    int diagonal_owner = i % size;
    if (rank == diagonal_owner) {
      int local_col = i / size;
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
    for (int j = 0; j < N; j++) {
      int element_owner = j % size;
      if (rank == element_owner) {
        if (i > j) {
          int local_column = j / size;
          for (int k = j; k < i; k++) {
            local_A_inv[i][local_column] +=
                fila[k] * local_A_inv[k][local_column];
          }
          local_A_inv[i][local_column] /= -fila[i];
        }
      }
    }
  }

  MPI_Datatype column, column_type;
  MPI_Type_vector(N, 1, N / size, MPI_DOUBLE, &column);
  MPI_Type_commit(&column);
  MPI_Type_create_resized(column, 0, 1 * sizeof(double), &column_type);
  MPI_Type_commit(&column_type);

  x = 0;
  for (int a = 0; a < N / size; a++) {
    double *recv_buffer = NULL;
    if (rank == ROOT) {
      recv_buffer = (double *)malloc(N * size * sizeof(double));
    }

    // Se devuelven las submatrices inversas calculadas al proceso root
    MPI_Gather(&(local_A_inv[0][a]), 1, column_type,
               (rank == ROOT ? recv_buffer : NULL), N, MPI_DOUBLE, ROOT,
               MPI_COMM_WORLD);
    if (rank == ROOT) {
      for (int j = 0; j < size; j++) { // Procesos
        for (int i = 0; i < N; i++) {  // Filas
          A[i][x + j] = recv_buffer[j * N + i];
        }
      }
      free(recv_buffer);
    }
    x += size;
  }
  // Impresion de la matriz inversa
  if (rank == ROOT) {
    printf("Matriz inversa:\n");
    imprimir_matriz(A, N, N);
    free_2d_double(&A);
  }

  free_2d_double(&local_A);
  free_2d_double(&local_A_inv);
  free(fila);

  MPI_Type_free(&block_type);

  MPI_Finalize();
  return 0;
}
