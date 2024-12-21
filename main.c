#include "/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define ROOT 0

// Reserva memoria dinámica para una matriz 2D con datos contiguos
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

// Libera la memoria dinámica de una matriz 2D
int free_2d_double(double ***array) {
  // free the memory of the first element of the array is at the start
  free(&((*array)[0][0]));

  // free the pointers into the memory
  free(*array);

  return 0;
}

/* Inicializa una matriz cuadrada como triangular inferior con valores
aleatorios */
void inicializar_matriz_triangular_inferior(double **matriz, int tamano) {
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
  printf("\n");
}

/*
 * La matriz triangular inferior es una matriz en la que todos los elementos por
 * encima de la diagonal principal son iguales a cero. Dada una matriz
 * triangular inferior L, su inversa L^-1 también es una matriz triangular
 * inferior. El cálculo de la inversa se realiza utilizando un procedimiento
 * iterativo basado en la siguiente fórmula:
 *
 * Para cada elemento L[i,j]^-1:
 *
 * 1. Si i == j (elemento diagonal), entonces:
 *  L[i, j]^-1 = 1 / L[i, j]
 *
 * 2. Si i > j (elemento debajo de la diagonal), entonces:
 *  L[i, j]^-1 = -1 / L[i, i] * (Sumatorio{k=j,k < i, k++} L[i, k] * L[k, j]^-1)
 */
void calcular_matriz_inversa(int N, int C, int rank, int size, double **A,
                             double **local_A, double **local_A_inv) {
  int bloques = N / C;
  // Calculo de la matriz inversa
  for (int row = 0; row < N; row++) {
    // Calcular el elemento de la diagonal
    int bloque_global = row / C;
    int diagonal_owner = bloque_global % size;
    if (rank == diagonal_owner) {
      int local_bloque = bloque_global / size;
      int local_col = (local_bloque * C) + (row % C);
      local_A_inv[row][local_col] = 1.0 / local_A[row][local_col];
    }

    double *current_row = (double *)malloc(N * sizeof(double));
    // Root difunde la fila actual a todos los procesos
    if (rank == ROOT) {
      for (int a = 0; a < N; a++) {
        current_row[a] = A[row][a];
      }
    }

    MPI_Bcast(current_row, N, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    // Calculo de los elementos que estan debajo de la diagonal
    for (int col = 0; col < N; col++) {
      int bloque_global = col / C;
      int element_owner = bloque_global % size;
      if (rank == element_owner) {
        if (row > col) {
          int local_bloque = bloque_global / size;
          int local_column = (local_bloque * C) + (col % C);
          for (int k = col; k < row; k++) {
            local_A_inv[row][local_column] +=
                current_row[k] * local_A_inv[k][local_column];
          }
          local_A_inv[row][local_column] /= -current_row[row];
        }
      }
    }
    free(current_row);
  }
}

int main(int argc, char *argv[]) {
  int rank, size, N, C;
  double **A = NULL;
  double **local_A = NULL;
  double **local_A_inv = NULL;
  MPI_Datatype block, global_block, local_block;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 3) {
    if (rank == ROOT) {
      printf("Uso: %s <N> <C>\n", argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  // Tamano de la matriz cuadrada
  N = atoi(argv[1]);
  // Tamano de los bloques de columnas
  C = atoi(argv[2]);

  // Inicializacion de la matriz completa
  if (rank == ROOT) {
    if ((N % (C * size)) != 0) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (malloc_2d_double(&A, N, N) == -1) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    inicializar_matriz_triangular_inferior(A, N);
    printf("Matriz inicial:\n");
    imprimir_matriz(A, N, N);
  }
  int bloques_totales = (N + C - 1) / C;
  int bloques_por_proceso = bloques_totales / size;

  // Reservar memoria dinámica para las submatrices locales
  if (malloc_2d_double(&local_A, N, bloques_por_proceso * C) == -1) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (malloc_2d_double(&local_A_inv, N, bloques_por_proceso * C) == -1) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N / size; j++) {
      local_A_inv[i][j] = 0.0;
    }
  }

  /* El paso entre bloques es el tamano de fila para obtener los
  elementos de las columnas, ya que no son contiguos */
  MPI_Type_vector(N, C, N, MPI_DOUBLE, &block);
  MPI_Type_commit(&block);
  // "Ocupa" C doubles para que el scatter envie el siguiente bloque
  MPI_Type_create_resized(block, 0, C * sizeof(double), &global_block);
  MPI_Type_commit(&global_block);

  MPI_Type_vector(N, C, bloques_por_proceso * C, MPI_DOUBLE, &local_block);
  MPI_Type_commit(&local_block);

  // Reparto de la matriz por columnas entre los procesos
  int recv_offset = 0;
  for (int send_column = 0; send_column < N; send_column += size * C) {
    MPI_Scatter((rank == ROOT ? &(A[0][send_column]) : NULL), 1, global_block,
                &(local_A[0][recv_offset]), 1, local_block, ROOT,
                MPI_COMM_WORLD);
    // Aumenta el offset para apuntar a la posicion del siguiente bloque
    recv_offset += C;
  }

  calcular_matriz_inversa(N, C, rank, size, A, local_A, local_A_inv);

  int col_offset = 0;
  for (int send_block = 0; send_block < bloques_por_proceso; send_block++) {
    // Se devuelven las submatrices inversas calculadas al proceso root
    MPI_Gather(&(local_A_inv[0][send_block * C]), 1, local_block,
               (rank == ROOT ? &(A[0][col_offset]) : NULL), 1, global_block,
               ROOT, MPI_COMM_WORLD);

    // Aumenta el offset para mover el puntero para los siguientes bloques
    col_offset += C * size;
  }

  // Impresion de la matriz inversa
  if (rank == ROOT) {
    printf("Matriz inversa:\n");
    imprimir_matriz(A, N, N);
    free_2d_double(&A);
  }

  free_2d_double(&local_A);
  free_2d_double(&local_A_inv);

  MPI_Type_free(&global_block);
  MPI_Type_free(&block);
  MPI_Type_free(&local_block);

  MPI_Finalize();
  return 0;
}
