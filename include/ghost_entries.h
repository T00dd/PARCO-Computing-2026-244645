#ifndef GHOST_ENTRIES_H
#define GHOST_ENTRIES_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct{
    int* csr_vector;
    int* csr_col;
    double* csr_val;

    int ghost_count;       
    int* ghost_indices;
}csr_matrix;

void identify_ghost_entries(csr_matrix* csr, int M_local, int N, int size, int rank);
void exchange_ghost_entries(const csr_matrix* csr, double* local_vector, double** ghost_vector_ptr, int N, int size, int rank, int local_N_start, int local_N_size);
int compare_ints(const void *a, const void *b);

#endif
