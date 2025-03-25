# MPI - Inversa Matriz Triangular

Este proyecto implementa un algoritmo paralelo utilizando MPI para calcular la inversa de una matriz triangular inferior. La distribución de la matriz se realiza de manera cíclica en bloques de columnas entre los procesos disponibles.

## Requisitos

Para compilar y ejecutar este programa, necesitas:

- Un sistema con soporte para MPI (Message Passing Interface)
- OpenMPI instalado
- Un compilador compatible con C (gcc recomendado)

## Compilación

Para compilar el código fuente, utiliza:

```sh
mpicc -o mpi_lower_tri_inv main.c
```

## Ejecución

El programa requiere que se especifiquen dos parámetros:

- `N`: Tamaño de la matriz cuadrada.
- `C`: Tamaño de los bloques de columnas a distribuir.

Para ejecutar el programa con `P` procesos, usa el siguiente comando:

```sh
mpirun -np P ./mpi_lower_tri_inv N C
```

Por ejemplo, para calcular la inversa de una matriz de tamaño 8x8 con bloques de 2 columnas usando 4 procesos:

```sh
mpirun -np 4 ./mpi_lower_tri_inv 8 2
```

## Funcionamiento

1. Se inicializa una matriz triangular inferior con valores aleatorios.
2. La matriz se distribuye entre los procesos en bloques de columnas de forma cíclica.
3. Se calcula la inversa de la matriz mediante un algoritmo iterativo paralelo:
   - Cada proceso calcula los elementos correspondientes a sus bloques asignados.
   - Se utiliza `MPI_Bcast` para compartir las filas necesarias entre los procesos.
   - Se emplea `MPI_Gather` para recolectar los resultados y reconstruir la matriz inversa en el proceso raíz.
4. Se imprime la matriz original y su inversa.

## Notas

- El tamaño de la matriz `N` debe ser divisible por `C * P`, de lo contrario, el programa abortará la ejecución.
- La matriz generada es siempre triangular inferior, por lo que la inversa también lo será.

## Ejemplo de Salida

```sh
Matriz inicial:
 1.00  0.00  0.00  0.00
 2.00  3.00  0.00  0.00
 4.00  5.00  6.00  0.00
 7.00  8.00  9.00 10.00

Matriz inversa:
 1.00  0.00  0.00  0.00
-0.67  0.33  0.00  0.00
 0.22 -0.28  0.17  0.00
-0.10  0.12 -0.15  0.10
```



