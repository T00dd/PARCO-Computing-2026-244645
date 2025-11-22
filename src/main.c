#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

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
int thread_num;
char* schedule_type;
int schedule_chunksize;
char* results = "../results/time_results.csv";

csr_matrix coo_to_csr(int M_, int nz_, int *I_, int* J_, double* val_);
double* vect_generator(int N_);
double multiplication(const csr_matrix*, const double* vector, int M_);
double multiplication_sequential(const csr_matrix* mat, const double* vector, int M_);

int main(int argc, char *argv[]){

    srand(time(NULL));

    if(argc < 5 || argc > 6){
        printf("Usage: %s [path of the matrix] [thread num] [schedule type] [schedule chunksize] [results .csv]\n", argv[0]);
        return -1;
    }

    matrix = argv[1];
    thread_num = atoi(argv[2]);
    schedule_type = argv[3];
    schedule_chunksize = atoi(argv[4]);

    if(argc == 6){
        results = argv[5];
    }

    if (thread_num < 1 || thread_num > 64){
        fprintf(stderr, "Invalid number of thread selected!\nPlease enter a number in this period [1,64]\n");
        return -1;
    }
    if (strcmp(schedule_type, "static") && strcmp(schedule_type, "dynamic") && strcmp(schedule_type, "guided")){
        fprintf(stderr, "Invalid schedule type selected!\nPlease enter one of this schedule type:\nstatic\ndynamic\nguided\n");
        return -1;
    }
    if (schedule_chunksize < 1 || schedule_chunksize > 10000){
        fprintf(stderr, "Invalid schedule chunksize selected!\nPlease enter a number in this period [1,10000]\n");
        return -1;
    }

    //setting enviroment variables:

    omp_set_num_threads(thread_num);
    
    if (strcmp(schedule_type, "static") == 0){
        omp_set_schedule(omp_sched_static, schedule_chunksize);
    }else{
        if (strcmp(schedule_type, "dynamic") == 0){
            omp_set_schedule(omp_sched_dynamic, schedule_chunksize);
        }else{
            omp_set_schedule(omp_sched_guided, schedule_chunksize);
        }        
    }    

    double *val;
    int *I, *J, M, N, nz;
    //I = indices rows, J = indices columns
    //M = numbers of rows, N = numbers of columns

    if (mm_read_unsymmetric_sparse(matrix, &M, &N, &nz, &val, &I, &J)){
        fprintf(stderr, "Unable to read the Matrix!\n");
        return -1;
    }
    
    csr_matrix csr = coo_to_csr(M, nz, I, J, val);

    double* random_vector = vect_generator(N);


    

    FILE* fp;

    fp = fopen(results, "a");
    if (fp == NULL){
        fprintf(stderr, "ERR: impossible to open %s\n", results);
        return -1;
    }

    int i;
    double time;

    if(argc == 6){

        time = multiplication(&csr, random_vector, M);
            
        if (time == -1.0) {
            fprintf(stderr, "Memory error, SpMV failed\n");
            return -1; 
        }
            
        if(thread_num == 1){
                fprintf(fp, "%s, %d, seq, NULL, %f\n", matrix, thread_num, time);
        }else{
                fprintf(fp, "%s, %d, %s, %d, %f\n", matrix, thread_num, schedule_type, schedule_chunksize, time);    
        }

    }else{

        for (i = 0; i < 10; i++){
        
            time = multiplication(&csr, random_vector, M);
            
            if (time == -1.0) {
                fprintf(stderr, "Memory error, SpMV failed\n");
                return -1; 
            }
            
            if(thread_num == 1){
                    fprintf(fp, "%s, %d, seq, NULL, %f\n", matrix, thread_num, schedule_type, schedule_chunksize, time);
            }else{
                    fprintf(fp, "%s, %d, %s, %d, %f\n", matrix, thread_num, schedule_type, schedule_chunksize, time);    
            }

        }
    }
    

    fclose(fp);

    free(I);
    free(J);
    free(val);

    free(csr.csr_col);
    free(csr.csr_val);
    free(csr.csr_vector);

    free(random_vector);

    return 0;
}

csr_matrix coo_to_csr(int M_, int nz_, int *I_, int* J_, double* val_){

    csr_matrix csr_mat;

    csr_mat.csr_col = malloc(nz_ * sizeof(int));
    csr_mat.csr_val = malloc(nz_ * sizeof(double));
    if (csr_mat.csr_col == NULL || csr_mat.csr_val == NULL) exit(1);

    csr_mat.csr_vector = calloc((M_ + 1), sizeof(int));
    if (csr_mat.csr_vector == NULL) exit(1);

	int i;
    //counting the nz elements for each row
    for(i = 0; i < nz_; i++){
        csr_mat.csr_vector[I_[i] + 1]++;
    }

    //prefix sum
    for (i = 1; i <= M_; i++){
        csr_mat.csr_vector[i] += csr_mat.csr_vector[i-1];
    }

    //reordering 
    int* row_pos = malloc((M_ + 1) * sizeof(int));
    if (row_pos == NULL) exit(1);
    memcpy(row_pos, csr_mat.csr_vector, (M_ + 1) * sizeof(int));

    for (i = 0; i < nz_; i++) {
        int row_index = I_[i];
        int dest_idx = row_pos[row_index];

        csr_mat.csr_col[dest_idx] = J_[i];
        csr_mat.csr_val[dest_idx] = val_[i];

        row_pos[row_index]++;
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

double multiplication(const csr_matrix* mat, const double* vector, int M_){

    if(thread_num == 1){
        return multiplication_sequential(mat, vector, M_);
    }

    double elapsed, finish, start;
    double* res_vect = malloc(M_ * sizeof(double));
    if (res_vect == NULL) {
        fprintf(stderr, "Errore di allocazione per il vettore risultato c.\n");
        return -1.0;
    }

    int i_;
    #pragma omp parallel for schedule(static)
    for (i_ = 0; i_ < M_; i_++){
        res_vect[i_] = 0.0;
    }
    

	int i, j;

    GET_TIME(start)
    #pragma omp parallel for default(none) shared(mat, vector, res_vect, M_) private(i, j) schedule(runtime)
    for(i = 0; i < M_; i++){

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

double multiplication_sequential(const csr_matrix* mat, const double* vector, int M_){

    double elapsed, finish, start;
    double* res_vect = malloc(M_ * sizeof(double));
    if (res_vect == NULL) {
        fprintf(stderr, "Errore di allocazione per il vettore risultato c.\n");
        return -1.0;
    }

	int i, j;

    for(i = 0; i < M_; i++){
        res_vect[i] = 0.0;
    }

    GET_TIME(start)
    for(i = 0; i < M_; i++){

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
