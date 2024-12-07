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
void calcular_matriz_inversa(int N, int rank, int size, double **A,
                             double **local_A, double **local_A_inv) {
  // Calculo de la matriz inversa
  for (int row = 0; row < N; row++) {
    // Calcular el elemento de la diagonal
    int diagonal_owner = row % size;
    if (rank == diagonal_owner) {
      int local_col = row / size;
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
      int element_owner = col % size;
      if (rank == element_owner) {
        if (row > col) {
          int local_column = col / size;
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
  double **A_inv = NULL;
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

    if (malloc_2d_double(&A_inv, N, N) == -1) {
      exit(EXIT_FAILURE);
    }

    inicializar_matriz_triangular_inferior(A, N);
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

  // Define un tipo derivado para representar los bloques de columnas
  MPI_Datatype block, block_type;
  /* El paso entre bloques es el tamano de fila para obtener los
  elementos de las columnas, ya que no son contiguos */
  MPI_Type_vector(N, 1, N, MPI_DOUBLE, &block);
  MPI_Type_commit(&block);
  // "Ocupa" 1 double para que el scatter envie la siguiente columna
  MPI_Type_create_resized(block, 0, 1 * sizeof(double), &block_type);
  MPI_Type_commit(&block_type);

  // Reparto de la matriz por columnas entre los procesos
  int recv_column = 0;
  for (int send_column = 0; send_column < N; send_column += size) {
    double *recv_buffer = (double *)malloc(N * sizeof(double));
    MPI_Scatter((rank == ROOT ? &(A[0][send_column]) : NULL), 1, block_type,
                recv_buffer, N, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    // Se copian los datos en la columna correspondiente de la matriz local
    for (int row = 0; row < N; row++) {
      local_A[row][recv_column] = recv_buffer[row];
    }
    // Avanza a la siguiente columna local
    recv_column++;
    free(recv_buffer);
  }

  // Se imprimen las submatrices
  printf("Submatriz local en el proceso %d:\n", rank);
  imprimir_matriz(local_A, N, N / size);
  printf("\n");
  MPI_Barrier(MPI_COMM_WORLD);

  calcular_matriz_inversa(N, rank, size, A, local_A, local_A_inv);

  // Tipo de dato para el envio de las columnas locales
  MPI_Datatype column, column_type;
  /* El paso entre bloques es el tamano de fila para obtener los
  elementos de las columnas, ya que no son contiguos */
  MPI_Type_vector(N, 1, N / size, MPI_DOUBLE, &column);
  MPI_Type_commit(&column);
  // "Ocupa" 1 double para que el gather envie la siguiente columna
  MPI_Type_create_resized(column, 0, 1 * sizeof(double), &column_type);
  MPI_Type_commit(&column_type);

  int col_offset = 0;
  for (int send_column = 0; send_column < N / size; send_column++) {
    double *recv_buffer = NULL;
    if (rank == ROOT) {
      recv_buffer = (double *)malloc(N * size * sizeof(double));
    }

    // Se devuelven las submatrices inversas calculadas al proceso root
    MPI_Gather(&(local_A_inv[0][send_column]), 1, column_type,
               (rank == ROOT ? recv_buffer : NULL), N, MPI_DOUBLE, ROOT,
               MPI_COMM_WORLD);

    if (rank == ROOT) {
      // Inserta las columnas recibidas en la matriz global
      for (int process = 0; process < size; process++) {
        for (int row = 0; row < N; row++) {
          A_inv[row][col_offset + process] = recv_buffer[process * N + row];
        }
      }
      free(recv_buffer);
    }
    // Actualiza la posicion de la columna
    col_offset += size;
  }

  // Impresion de la matriz inversa
  if (rank == ROOT) {
    printf("Matriz inversa:\n");
    imprimir_matriz(A_inv, N, N);
    free_2d_double(&A);
    free_2d_double(&A_inv);
  }

  free_2d_double(&local_A);
  free_2d_double(&local_A_inv);

  MPI_Type_free(&block_type);
  MPI_Type_free(&block);
  MPI_Type_free(&column_type);
  MPI_Type_free(&column);

  MPI_Finalize();
  return 0;
}
