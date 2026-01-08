#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#include "mmio.h"
#include "ghost_entries.h"

#define MAX_RAND 10000
#define MIN_RAND -10000

//global variables

char* matrix = NULL;
char* scaling_type = NULL;
char* parallelization = NULL;
char* results_ss = "../results/time_results_strong.csv";
char* results_ws = "../results/time_results_weak.csv";


csr_matrix coo_to_csr(int M_local, int nz_, int *I_, int* J_, double* val_, int size, int rank, int M_total);
double* vect_generator(int N_);
double multiplication(const csr_matrix* mat, const double* extended_column_vect, int M_local);
int compare_doubles(const void *a, const void *b);


int main(int argc, char *argv[]){

    srand(time(NULL));

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc != 3){
        if(rank == 0){
            printf("Usage: mpirun -n num_proc %s [path of the matrix] [type of scaling]\n", argv[0]);
            printf("'-ws' for weak scaling, '-ss' for strong scaling\n");
        }
        MPI_Finalize();
        return -1;
    }

    matrix = argv[1];
    scaling_type = argv[2];
    
    if(strcmp(scaling_type, "-ss") && strcmp(scaling_type, "-ws")){
        if(rank == 0){
            printf("Scaling type not accepted! Select between:\nWeak scaling (-ws)\nStrong scaling (-ss)\n");
        }
        MPI_Finalize();
        return -1;
    }

    double *val = NULL;
    int *I = NULL, *J = NULL;
    int M, N, nz;
    //I = indices rows, J = indices columns
    //M = numbers of rows, N = numbers of columns
    
    if(rank == 0){ 
        if(mm_read_unsymmetric_sparse(matrix, &M, &N, &nz, &val, &I, &J)){
            fprintf(stderr, "Unable to read the Matrix!\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
    
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        printf("Matrix dimensions: M=%d, N=%d, nz=%d\n", M, N, nz);
    }

    int *send_counts = calloc(size, sizeof(int));

    if(rank == 0){
        int k;
        for(k = 0; k < nz; k++){
            int target_rank = I[k] % size;
            send_counts[target_rank]++;
        }   
    }

    //telling each rank how many nz values it will receive
    int local_nz;
    MPI_Scatter(send_counts, 1, MPI_INT, &local_nz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *local_I = malloc(local_nz * sizeof(int));
    int *local_J = malloc(local_nz * sizeof(int));
    double *local_Val = malloc(local_nz * sizeof(double));

    if(rank == 0){

        int **temp_I = malloc(size * sizeof(int*));
        int **temp_J = malloc(size * sizeof(int*));
        double **temp_Val = malloc(size * sizeof(double*));
        int *current_position = calloc(size, sizeof(int));

        int p;
        for(p = 0; p < size; p++){

            temp_I[p] = malloc(send_counts[p]*sizeof(int));
            temp_J[p] = malloc(send_counts[p]*sizeof(int));
            temp_Val[p] = malloc(send_counts[p]*sizeof(double));

        }

        int k;
        for(k = 0; k < nz; k++){
            int p = I[k] % size; 
            
            int pos = current_position[p];
            temp_I[p][pos] = I[k];
            temp_J[p][pos] = J[k];
            temp_Val[p][pos] = val[k];
            current_position[p]++;
        }


        for(p = 1; p < size; p++){
            MPI_Send(temp_I[p], send_counts[p], MPI_INT, p, 0, MPI_COMM_WORLD);
            MPI_Send(temp_J[p], send_counts[p], MPI_INT, p, 1, MPI_COMM_WORLD);
            MPI_Send(temp_Val[p], send_counts[p], MPI_DOUBLE, p, 2, MPI_COMM_WORLD);
        }

        memcpy(local_I, temp_I[0], send_counts[0]*sizeof(int));
        memcpy(local_J, temp_J[0], send_counts[0]*sizeof(int));
        memcpy(local_Val, temp_Val[0], send_counts[0]*sizeof(double));

        for(p = 0; p< size; p++){
            free(temp_I[p]);
            free(temp_J[p]);
            free(temp_Val[p]);
        }

        free(temp_I);
        free(temp_J);
        free(temp_Val);
        free(current_position);

    }else{

        MPI_Recv(local_I, local_nz, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_J, local_nz, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_Val, local_nz, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    //each rank call the function
    int M_local = M / size;
    if(rank < (M % size)) {
        M_local++;
    }    

    int local_N_size = N / size;
    if(rank < (N % size)) {
        local_N_size++;
    }

    csr_matrix csr = coo_to_csr(M_local, local_nz, local_I, local_J, local_Val, size, rank, M);

    //generate the complete vector only on rank 0
    double *global_random_vector = NULL;
    if(rank ==  0){
        global_random_vector = vect_generator(N);
    }
    
    //allocating space for the random vector on all the ranks
    double *local_random_vector = malloc(local_N_size * sizeof(double));

    //distribution of th erandom vector
    if(rank == 0){
        int local_idx = 0;
        int k;
        for(k = 0; k < N; k++){
            int target_rank = k % size;
            
            if(target_rank == 0){
                local_random_vector[local_idx++] = global_random_vector[k];
            } else {
                MPI_Send(&global_random_vector[k], 1, MPI_DOUBLE, target_rank, k, MPI_COMM_WORLD);
            }
        }

        free(global_random_vector);

    } else {
        int k;
        for(k = 0; k < local_N_size; k++){
            int global_idx = rank + k * size;
            MPI_Recv(&local_random_vector[k], 1, MPI_DOUBLE, 0, global_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }



    identify_ghost_entries(&csr, M_local, N, size, rank);
    renumber_column_indices(&csr, M_local, local_N_size, size, rank);

    int extended_size = local_N_size + csr.ghost_count;
    double* extended_column_vect = malloc(extended_size * sizeof(double));

    //copiamo solo la parte locale
    memcpy(extended_column_vect, local_random_vector, local_N_size * sizeof(double));


    double local_time_multiplication, global_time_comunication, global_time_multiplication;
    double all_times_com[10];
    double all_times_mult[10];
    double *ghost_vector = NULL;
    int i;
    for(i = 0; i<10; i++){

        MPI_Barrier(MPI_COMM_WORLD);

        //TIME FOR COMUNICATION
        double start_local_time_comunication = MPI_Wtime();
        exchange_ghost_entries(&csr, local_random_vector, &ghost_vector, N, size, rank, local_N_size);
        if(csr.ghost_count > 0){//cpiamo anche la parte ghost
            memcpy(extended_column_vect + local_N_size, ghost_vector, csr.ghost_count * sizeof(double));
        }
        double end_local_time_comunication =MPI_Wtime();
        double local_time_comunication = end_local_time_comunication - start_local_time_comunication;


        //TIME FOR MULTIPLICATION
        local_time_multiplication = multiplication(&csr, extended_column_vect, M_local);            
        
        MPI_Reduce(&local_time_comunication, &global_time_comunication, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_time_multiplication, &global_time_multiplication, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if(rank==0){
            all_times_com[i] = global_time_comunication;
            all_times_mult[i] = global_time_multiplication;
        }

        free(ghost_vector);
        ghost_vector = NULL;

    }

    if(rank == 0){

        FILE* fp;

        if(strcmp(scaling_type, "-ws")){
            fp = fopen(results_ss, "a");  
            
            if (fp == NULL){
                fprintf(stderr, "ERR: impossible to open %s\n", results_ss);
                MPI_Abort(MPI_COMM_WORLD, -1);
                return -1;
            }
        }else{
            fp = fopen(results_ws, "a"); 
            
            if (fp == NULL){
                fprintf(stderr, "ERR: impossible to open %s\n", results_ws);
                MPI_Abort(MPI_COMM_WORLD, -1);
                return -1;
            }
        }
        
        qsort(all_times_com, 10, sizeof(double), compare_doubles);
        qsort(all_times_mult, 10, sizeof(double), compare_doubles);
        double percentile_90_com = all_times_com[8];
        double percentile_90_mult = all_times_mult[8];

        double ms_com = percentile_90_com * 1000.0;
        double ms_mult = percentile_90_mult;
        double total_flops = (2.0 * nz) / (ms_mult/1000.0);
        double gflops = total_flops / 1e9;

        printf("Prestazioni: %f GFLOPS\n", gflops);

        fprintf(fp, "%s, %d, %f, %f, %f\n", matrix, size, percentile_90_com, percentile_90_mult, gflops);

        fclose(fp);


        free(I);
        free(J);
        free(val);

    }
    
    free(extended_column_vect);
    free(local_I);
    free(local_J);
    free(local_Val);
    free(local_random_vector);
    free(send_counts);

    free(csr.csr_col);
    free(csr.csr_val);
    free(csr.csr_vector);
    free(csr.ghost_indices);

    MPI_Finalize();
    return 0;
}

csr_matrix coo_to_csr(int M_local, int nz_, int *I_, int* J_, double* val_, int size, int rank, int M_total){
    csr_matrix csr_mat;
    
    csr_mat.csr_col = malloc(nz_ * sizeof(int));
    csr_mat.csr_val = malloc(nz_ * sizeof(double));
    if (csr_mat.csr_col == NULL || csr_mat.csr_val == NULL) exit(1);
    
    csr_mat.csr_vector = calloc((M_local + 1), sizeof(int));
    if (csr_mat.csr_vector == NULL) exit(1);
    
    int i;
    for(i = 0; i < nz_; i++){
        int global_row = I_[i];
        int local_row = global_row / size; 
        
        if(local_row < 0 || local_row >= M_local) {
            fprintf(stderr, "Rank %d: ERROR! global_row=%d -> local_row=%d out of bounds [0,%d)\n", rank, global_row, local_row, M_local);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        
        csr_mat.csr_vector[local_row + 1]++;
    }
    
    //prefix sum 
    for (i = 1; i <= M_local; i++){
        csr_mat.csr_vector[i] += csr_mat.csr_vector[i-1];
    }
    
    //reordering
    int* row_pos = malloc((M_local + 1) * sizeof(int));
    if (row_pos == NULL) exit(1);
    memcpy(row_pos, csr_mat.csr_vector, (M_local + 1) * sizeof(int));
    
    for (i = 0; i < nz_; i++) {
        int global_row = I_[i];
        int local_row = global_row / size;
        int dest_idx = row_pos[local_row];
        
        csr_mat.csr_col[dest_idx] = J_[i];
        csr_mat.csr_val[dest_idx] = val_[i];
        
        row_pos[local_row]++;
    }
    
    free(row_pos);
    
    csr_mat.ghost_count = 0;
    csr_mat.ghost_indices = NULL;
    
    return csr_mat;
}



double* vect_generator(int N_){
    double* vect = malloc(N_ * sizeof(double));
    int i;
    for (i = 0; i < N_; i++){ vect[i] = (double) (rand()% (MAX_RAND - MIN_RAND + 1) + MIN_RAND) / 1000.0; }

    return vect;    
}

double multiplication(const csr_matrix* mat, const double* extended_column_vect, int M_local){

    double elapsed, finish, start;
    double* res_vect = malloc(M_local * sizeof(double));
    if (res_vect == NULL) {
        fprintf(stderr, "Errore di allocazione per il vettore risultato c.\n");
        return -1.0;
    }

    int i;
    for (i = 0; i < M_local; i++){
        res_vect[i] = 0.0;
    }

    start = MPI_Wtime();
    
    for(i = 0; i < M_local; i++){
        double sum = 0.0;
        int j;
        for(j = mat->csr_vector[i]; j < mat->csr_vector[i + 1]; j++){
            sum += mat->csr_val[j] * extended_column_vect[mat->csr_col[j]];
        }

        res_vect[i] = sum;
    }
    finish = MPI_Wtime();

    elapsed = finish - start;
    free(res_vect);

    return (elapsed * 1000);
}

int compare_doubles(const void *a, const void *b) {
    double arg1 = *(const double *)a;
    double arg2 = *(const double *)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}
