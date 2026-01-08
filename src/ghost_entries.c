#include "ghost_entries.h"

void identify_ghost_entries(csr_matrix* csr, int M_local, int N, int size, int rank){
    
    
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
    csr->ghost_indices = malloc(unique_count * sizeof(int));
    
    for(i = 0; i < unique_count; i++){
        if(temp_cols[i] % size != rank){
            csr->ghost_indices[csr->ghost_count] = temp_cols[i];
            csr->ghost_count++;
        }
    }
    
    //we reduce the allocated memory to the right lenght
    if(csr->ghost_count > 0) {
        csr->ghost_indices = realloc(csr->ghost_indices, csr->ghost_count * sizeof(int));
    } else {
        free(csr->ghost_indices);
        csr->ghost_indices = NULL;
    }
        
    free(temp_cols);
}

void renumber_column_indices(csr_matrix* csr, int M_local, int local_N_size, int size, int rank) {
    
    //renumbering every column indices on csr matrix
    int i, j;
    for(i = 0; i < M_local; i++){
        for(j = csr->csr_vector[i]; j < csr->csr_vector[i+1]; j++){
            int global_col = csr->csr_col[j];
            
            //local column
            if(global_col % size == rank){
                // Mappa a [0, local_N_size)
                csr->csr_col[j] = global_col / size;
            }else{//ghost column

                //binary search in ghost_indices (already ordered)
                int left = 0;
                int right = csr->ghost_count - 1;
                int ghost_pos = -1;
                
                while(left <= right){
                    int mid = left + (right - left) / 2;
                    if(csr->ghost_indices[mid] == global_col){
                        ghost_pos = mid;
                        break;
                    }else if(csr->ghost_indices[mid] < global_col){
                        left = mid + 1;
                    }else{
                        right = mid - 1;
                    }
                }
                

                //mapping [local_N_size, local_N_size + ghost_count)
                if(ghost_pos >= 0){
                    csr->csr_col[j] = local_N_size + ghost_pos;
                } else {
                    fprintf(stderr, "ERROR: Ghost index %d not found in ghost_indices!\n", global_col);
                }
            }
        }
    }
}


void exchange_ghost_entries(const csr_matrix* csr, double* local_vector, double** ghost_vector_ptr, int N, int size, int rank, int local_N_size){
    
    //space allocation for ghosts vectors
    *ghost_vector_ptr = malloc(csr->ghost_count * sizeof(double));
    if(*ghost_vector_ptr == NULL && csr->ghost_count > 0){
        fprintf(stderr, "Errore allocazione ghost_vector\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    

    //send_counts[i] = how many indices i need to send (how many ghosts i want)
    //recv_counts[i] = how many indices i receive from that rank (how many ghosts that rank wants from me)
    int* send_counts = calloc(size, sizeof(int));
    int* recv_counts = calloc(size, sizeof(int));
    
    int base_cols_per_rank = N / size;
    int remainder = N % size;
    
    //counting how many ghosts i want from each rank
    int i;
    for(i = 0; i < csr->ghost_count; i++){
        int global_col = csr->ghost_indices[i];
        int owner_rank = global_col % size;
        
        if(owner_rank < 0 || owner_rank >= size) {
            fprintf(stderr, "Rank %d: ERROR! global_col=%d -> owner_rank=%d invalid\n",
                    rank, global_col, owner_rank);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        
        send_counts[owner_rank]++;
    }
    
    //we twll all the ranks how many  values we need
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);    
    
    //initialising buffers
    int total_send = 0, total_recv = 0;
    for(i = 0; i < size; i++){
        total_send += send_counts[i];
        total_recv += recv_counts[i];
    }
    
    
    int* indices_i_request = malloc(total_send * sizeof(int));
    int* indices_others_request = malloc(total_recv * sizeof(int));
    double* values_i_receive = malloc(total_send * sizeof(double));
    double* values_i_send = malloc(total_recv * sizeof(double));
    

    MPI_Request* requests = malloc(2 * size * sizeof(MPI_Request));
    int req_count = 0;
    
    int* recv_offsets = malloc(size * sizeof(int));
    recv_offsets[0] = 0;
    for(i = 1; i < size; i++){
        recv_offsets[i] = recv_offsets[i-1] + recv_counts[i-1];
    }
    
    //we receive the indices that other ranks ask for    
    for(i = 0; i < size; i++){
        if(recv_counts[i] > 0){
            MPI_Irecv(&indices_others_request[recv_offsets[i]], recv_counts[i], MPI_INT, i, 0, MPI_COMM_WORLD, &requests[req_count++]);
        }
    }
    
     //we prepare the indices that this rank need
    int* send_offsets = malloc(size * sizeof(int));
    send_offsets[0] = 0;
    for(i = 1; i < size; i++){
        send_offsets[i] = send_offsets[i-1] + send_counts[i-1];
    }
    

    int* current_send_pos = calloc(size, sizeof(int));
    for(i = 0; i < csr->ghost_count; i++){
        int global_col = csr->ghost_indices[i];
        int owner_rank = global_col % size;  // MODULO
        
        int pos = send_offsets[owner_rank] + current_send_pos[owner_rank];
        indices_i_request[pos] = global_col;
        current_send_pos[owner_rank]++;
    }
    
    //we send the indices
    for(i = 0; i < size; i++){
        if(send_counts[i] > 0){
            MPI_Isend(&indices_i_request[send_offsets[i]], send_counts[i], MPI_INT, i, 0, MPI_COMM_WORLD, &requests[req_count++]);
        }
    }
    
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    
    //we prepare the values to send
    for(i = 0; i < total_recv; i++){
        int global_idx = indices_others_request[i];
        
        if(global_idx < 0 || global_idx >= N) {
            fprintf(stderr, "Rank %d: CORRUPTED request[%d] = %d (N=%d)\n",
                    rank, i, global_idx, N);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        
        int local_idx = global_idx / size;
        
        if(local_idx < 0 || local_idx >= local_N_size) {
            fprintf(stderr, "Rank %d: ERROR! Received request for col %d → local_idx %d out of [0,%d)\n",
                    rank, global_idx, local_idx, local_N_size);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        
        values_i_send[i] = local_vector[local_idx];
    }
    
    //we exchange the values
    req_count = 0;
    
    for(i = 0; i < size; i++){
        if(send_counts[i] > 0){
            MPI_Irecv(&values_i_receive[send_offsets[i]], send_counts[i], MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &requests[req_count++]);
        }
    }
    
    for(i = 0; i < size; i++){
        if(recv_counts[i] > 0){
            MPI_Isend(&values_i_send[recv_offsets[i]], recv_counts[i], MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &requests[req_count++]);
        }
    }
    
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    
    //we copy the values in the final array (ghost_vector)
    memset(current_send_pos, 0, size * sizeof(int));
    for(i = 0; i < csr->ghost_count; i++){
        int global_col = csr->ghost_indices[i];
        int owner_rank = global_col % size;  // MODULO
        
        int pos = send_offsets[owner_rank] + current_send_pos[owner_rank];
        (*ghost_vector_ptr)[i] = values_i_receive[pos];
        current_send_pos[owner_rank]++;
    }
    
    free(send_counts);
    free(recv_counts);
    free(indices_i_request);
    free(indices_others_request);
    free(values_i_receive);
    free(values_i_send);
    free(requests);
    free(send_offsets);
    free(recv_offsets);
    free(current_send_pos);
}



int compare_ints(const void *a, const void *b){
    return (*(int*)a - *(int*)b);
}


