#include "ghost_entries.h"

void identify_ghost_entries(csr_matrix* csr, int M_local, int N, int size, int rank,  int local_N_start, int local_N_size){
    
    
    //we allocate a temporal array for all the indices of the columns
    int* temp_cols = malloc(csr->csr_vector[M_local] * sizeof(int));
    int temp_count = 0;
    
    int i, j;

    for(i = 0; i < M_local; i++){
        for(j = csr->csr_vector[i]; j < csr->csr_vector[i + 1]; j++){
            temp_cols[temp_count++] = csr->csr_col[j]; //we obtain the global index for this nz value
        }
    }

    //then we need to verify if this column has already been registered
    qsort(temp_cols, temp_count, sizeof(int), compare_ints);
    int unique_count = 0;
    if(temp_count > 0){

        temp_cols[unique_count++] = temp_cols[0];
        
        for(i = 1; i < temp_count; i++) {

            if(temp_cols[i] != temp_cols[i-1]) {
                temp_cols[unique_count++] = temp_cols[i];
            }
        }
    }

    

    csr->ghost_count = 0;
    csr->ghost_indices = malloc(temp_count * sizeof(int));
    csr->local_to_global = malloc(temp_count * sizeof(int));
    
    int local_count = 0;
    for(i = 0; i < temp_count; i++){
        if(temp_cols[i] < local_N_start || temp_cols[i] >= (local_N_start + local_N_size)) {
            //if outside of this range it's a ghost entry
            csr->ghost_indices[csr->ghost_count] = temp_cols[i];
            csr->ghost_count++;
        }
    }
    
    //we reduce the allocated memory to the right lenght
    csr->ghost_indices = realloc(csr->ghost_indices, csr->ghost_count * sizeof(int));
    
    free(temp_cols);
}

void exchange_ghost_entries(const csr_matrix* csr, double* local_vector, double** ghost_vector_ptr, int N, int size, int rank, int local_N_start, int local_N_size){
    
    //space allocation for ghosts vectors
    *ghost_vector_ptr = malloc(csr->ghost_count * sizeof(double));
    if(*ghost_vector_ptr == NULL && csr->ghost_count > 0){
        fprintf(stderr, "Errore allocazione ghost_vector\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    //how many values the rank has to receive and send
    int* send_counts = calloc(size, sizeof(int));
    int* recv_counts = calloc(size, sizeof(int));
    
    
    //we determine which rank has the column
    for(int i = 0; i < csr->ghost_count; i++){
        int global_col = csr->ghost_indices[i];
        int owner_rank = global_col / ((N / size) + 1);
        
        //adjusting non uniform distribution
        if(owner_rank >= size) owner_rank = size - 1;
        int owner_start = owner_rank * (N / size) + (owner_rank < (N % size) ? owner_rank : (N % size));
        
        while(global_col < owner_start && owner_rank > 0){
            owner_rank--;
            owner_start = owner_rank * (N / size) + (owner_rank < (N % size) ? owner_rank : (N % size));
        }
        
        recv_counts[owner_rank]++;
    }
    
    //we twll all the ranks how many  values we need
    MPI_Alltoall(recv_counts, 1, MPI_INT, send_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
    //initialising buffers
    int total_send = 0, total_recv = 0;
    for(int i = 0; i < size; i++){
        total_send += send_counts[i];
        total_recv += recv_counts[i];
    }
    
    int* send_indices = malloc(total_send * sizeof(int));
    double* send_values = malloc(total_send * sizeof(double));
    int* recv_indices = malloc(total_recv * sizeof(int));
    double* recv_values = malloc(total_recv * sizeof(double));
    

    MPI_Request* requests = malloc(2 * size * sizeof(MPI_Request));
    int req_count = 0;
    
    int* recv_offsets = malloc(size * sizeof(int));
    recv_offsets[0] = 0;
    for(int i = 1; i < size; i++){
        recv_offsets[i] = recv_offsets[i-1] + recv_counts[i-1];
    }
    
    //we receive the indices that other ranks ask for
    for(int i = 0; i < size; i++){
        if(recv_counts[i] > 0){
            MPI_Irecv(&recv_indices[recv_offsets[i]], recv_counts[i], MPI_INT, i, 0, MPI_COMM_WORLD, &requests[req_count++]);
        }
    }
    
    //we prepare the indices that this rank need
    int* send_offsets = malloc(size * sizeof(int));
    send_offsets[0] = 0;
    for(int i = 1; i < size; i++){
        send_offsets[i] = send_offsets[i-1] + send_counts[i-1];
    }
    
    int* current_send_pos = calloc(size, sizeof(int));
    for(int i = 0; i < csr->ghost_count; i++){
        int global_col = csr->ghost_indices[i];
        int owner_rank = global_col / ((N / size) + 1);
        if(owner_rank >= size) owner_rank = size - 1;
        
        int pos = send_offsets[owner_rank] + current_send_pos[owner_rank];
        send_indices[pos] = global_col;
        current_send_pos[owner_rank]++;
    }
    
    //we send the indices
    for(int i = 0; i < size; i++){
        if(send_counts[i] > 0){
            MPI_Isend(&send_indices[send_offsets[i]], send_counts[i], MPI_INT, i, 0, MPI_COMM_WORLD, &requests[req_count++]);
        }
    }
    
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    
    //we prepare the values to send
    for(int i = 0; i < total_recv; i++){
        int global_idx = recv_indices[i];
        int local_idx = global_idx - local_N_start;
        recv_values[i] = local_vector[local_idx];
    }
    
    //we exchange the values
    req_count = 0;
    for(int i = 0; i < size; i++){
        if(send_counts[i] > 0){
            MPI_Irecv(&send_values[send_offsets[i]], send_counts[i], MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &requests[req_count++]);
        }
    }
    
    for(int i = 0; i < size; i++){
        if(recv_counts[i] > 0){
            MPI_Isend(&recv_values[recv_offsets[i]], recv_counts[i], MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &requests[req_count++]);
        }
    }
    
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    
    //we copy the values in the final array (ghost_vector)
    memset(current_send_pos, 0, size * sizeof(int));
    for(int i = 0; i < csr->ghost_count; i++){
        int global_col = csr->ghost_indices[i];
        int owner_rank = global_col / ((N / size) + 1);
        if(owner_rank >= size) owner_rank = size - 1;
        
        int pos = send_offsets[owner_rank] + current_send_pos[owner_rank];
        (*ghost_vector_ptr)[i] = send_values[pos];
        current_send_pos[owner_rank]++;
    }
    
    free(send_counts);
    free(recv_counts);
    free(send_indices);
    free(send_values);
    free(recv_indices);
    free(recv_values);
    free(requests);
    free(send_offsets);
    free(recv_offsets);
    free(current_send_pos);
}

int compare_ints(const void *a, const void *b){
    return (*(int*)a - *(int*)b);
}


