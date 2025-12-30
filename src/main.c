#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#include "timer.h"
#include "mmio.h"

#define MAX_RAND 10000
#define MIN_RAND -10000

//global variables

typedef struct{
    int* csr_vector;
    int* csr_col;
    double* csr_val;
}csr_matrix;

char* matrix;
char* scaling_type;
char* results = "../results/time_results.csv";

csr_matrix coo_to_csr(int M_local, int nz_, int *I_, int* J_, double* val_, int size, int rank);
double* vect_generator(int N_);
double multiplication(const csr_matrix*, const double* vector, int M_local);
double multiplication_sequential(const csr_matrix* mat, const double* vector, int M_local);
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
            printf("-ws for weak scaling, -ss for strong scaling\n");
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
    int M_local = (M / size) + (rank < (M % size) ? 1 : 0);    
    csr_matrix csr = coo_to_csr(M_local, local_nz, local_I, local_J, local_Val, size, rank);

    double* random_vector = NULL
    if(rank ==  0){
        random_vector = vect_generator(N);
    }else{
        random_vector = malloc(N * sizeof(double));
    }

    MPI_Bcast(random_vector, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double local_time, max_time;
    double all_times[10];
    int i;
    for(i = 0; i<10; i++){

        MPI_Barrier(MPI_COMM_WORLD);
        local_time = multiplication(&csr, random_vector, M_local);
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if(rank==0){
            all_times[i] = max_time;
        }

    }

    if(rank == 0){

        FILE* fp;

        fp = fopen(results, "a");
        if (fp == NULL){
            fprintf(stderr, "ERR: impossible to open %s\n", results);
            MPI_Abort(MPI_COMM_WORLD, -1);
            return -1;
        }

        qsort(all_times, 10, sizeof(double), compare_doubles);
        double percentile_90 = all_times[8];

        double seconds = percentile_90 / 1000.0;
        double total_flops = (2.0 * nz) / seconds;
        double mflops = total_flops / 1e6;
        double gflops = total_flops / 1e9;

        printf("Prestazioni: %f GFLOPS\n", gflops);

        fprintf(fp, "%s, %d, %f, %f\n", matrix, size, percentile_90, mflops);

        fclose(fp);


        free(I);
        free(J);
        free(val);

        free(csr.csr_col);
        free(csr.csr_val);
        free(csr.csr_vector);

        free(random_vector);
    }

    

    // int i;
    // double time;

    // if(argc == 6){

    //     time = multiplication(&csr, random_vector, M);
            
    //     if (time == -1.0) {
    //         fprintf(stderr, "Memory error, SpMV failed\n");
    //         MPI_Finalize();
    //         return -1; 
    //     }
            
    //     if(thread_num == 1){
    //             fprintf(fp, "%s, %d, seq, NULL, %f\n", matrix, thread_num, time);
    //     }else{
    //             fprintf(fp, "%s, %d, %s, %d, %f\n", matrix, thread_num, schedule_type, schedule_chunksize, time);    
    //     }

    // }else{

    //     for (i = 0; i < 10; i++){
        
    //         time = multiplication(&csr, random_vector, M);
            
    //         if (time == -1.0) {
    //             fprintf(stderr, "Memory error, SpMV failed\n");
    //             MPI_Finalize();
    //             return -1; 
    //         }
            
    //         if(thread_num == 1){
    //                 fprintf(fp, "%s, %d, seq, NULL, %f\n", matrix, thread_num, time);
    //         }else{
    //                 fprintf(fp, "%s, %d, %s, %d, %f\n", matrix, thread_num, schedule_type, schedule_chunksize, time);    
    //         }

    //     }
    // }
    

    

    MPI_Finalize();
    return 0;
}

csr_matrix coo_to_csr(int M_local, int nz_, int *I_, int* J_, double* val_, int size, int rank){

    csr_matrix csr_mat;

    csr_mat.csr_col = malloc(nz_ * sizeof(int));
    csr_mat.csr_val = malloc(nz_ * sizeof(double));
    if (csr_mat.csr_col == NULL || csr_mat.csr_val == NULL) exit(1);

    csr_mat.csr_vector = calloc((M_local + 1), sizeof(int));
    if (csr_mat.csr_vector == NULL) exit(1);

	int i;
    //counting the nz elements for each row
    for(i = 0; i < nz_; i++){
        int local_row = I_[i] % size;
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
        int local_row = I_[i] % size;
        int dest_idx = row_pos[local_row];

        csr_mat.csr_col[dest_idx] = J_[i];
        csr_mat.csr_val[dest_idx] = val_[i];

        row_pos[local_row]++;
    }

    free(row_pos);

    return csr_mat;
}

double* vect_generator(int N_){
    double* vect = malloc(N_ * sizeof(double));
    int i;
    for (i = 0; i < N_; i++){ vect[i] = (double) (rand()% (MAX_RAND - MIN_RAND + 1) + MIN_RAND) / 1000.0; }

    return vect;    
}

double multiplication(const csr_matrix* mat, const double* vector, int M_local){

    if(thread_num == 1){
        return multiplication_sequential(mat, vector, M_local);
    }

    double elapsed, finish, start;
    double* res_vect = malloc(M_local * sizeof(double));
    if (res_vect == NULL) {
        fprintf(stderr, "Errore di allocazione per il vettore risultato c.\n");
        return -1.0;
    }

    int i, j;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < M_local; i++){
        res_vect[i] = 0.0;
    }
    

    GET_TIME(start)
    #pragma omp parallel for default(none) shared(mat, vector, res_vect, M_local) private(i, j) schedule(runtime)
    for(i = 0; i < M_local; i++){

        double sum = 0.0;

        for(j = mat->csr_vector[i]; j < mat->csr_vector[i + 1]; j++){
            sum += mat->csr_val[j] * vector[mat->csr_col[j]];
        }

        res_vect[i] = sum;
    }
    GET_TIME(finish)

    elapsed = finish - start;
    free(res_vect);

    return (elapsed * 1000);
}

double multiplication_sequential(const csr_matrix* mat, const double* vector, int M_local){

    double elapsed, finish, start;
    double* res_vect = malloc(M_local * sizeof(double));
    if (res_vect == NULL) {
        fprintf(stderr, "Errore di allocazione per il vettore risultato c.\n");
        return -1.0;
    }

	int i, j;

    for(i = 0; i < M_local; i++){
        res_vect[i] = 0.0;
    }

    GET_TIME(start)
    for(i = 0; i < M_local; i++){

        double sum = 0.0;

        for(j = mat->csr_vector[i]; j < mat->csr_vector[i + 1]; j++){
            sum += mat->csr_val[j] * vector[mat->csr_col[j]];
        }

        res_vect[i] = sum;
    }
    GET_TIME(finish)

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
