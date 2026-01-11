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
            if(temp_cols[i] != temp_cols[i-1]){
                temp_cols[unique_count++] = temp_cols[i];
            }
        }
    }
    

    csr->ghost_count = 0;
    csr->ghost_indices = malloc(unique_count * sizeof(int));
    
    for(i = 0; i < unique_count; i++){
        if(temp_cols[i] % size != rank){
            csr->ghost_indices[csr->ghost_count++] = temp_cols[i];
        }
    }
    
    //we reduce the allocated memory to the right lenght
    if(csr->ghost_count > 0) {
        csr->ghost_indices = realloc(csr->ghost_indices, csr->ghost_count * sizeof(int));
    }else{
        free(csr->ghost_indices);
        csr->ghost_indices = NULL;
    }

    //we create the map ghost_to_local for faster access time
    int max_global_col = 0;
    for(i = 0; i < unique_count; i++){
        if(temp_cols[i] > max_global_col){
            max_global_col = temp_cols[i];
        }
    }

    csr->ghost_to_local = malloc((max_global_col + 1) * sizeof(int));
    //-1 means it's not a ghost
    for(i = 0; i <= max_global_col; i++){
        csr->ghost_to_local[i] = -1;
    }

    for(i = 0; i < csr->ghost_count; i++){
        int global_col = csr->ghost_indices[i];
        csr->ghost_to_local[global_col] = i;
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

                int ghost_pos = csr->ghost_to_local[global_col];
                

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
    
    //counting how many ghosts i want from each rank
    int i;
    for(i = 0; i < csr->ghost_count; i++){
        int global_col = csr->ghost_indices[i];
        int owner_rank = global_col % size;
        
        if(owner_rank < 0 || owner_rank >= size) {
            fprintf(stderr, "Rank %d: ERROR! global_col=%d -> owner_rank=%d invalid\n", rank, global_col, owner_rank);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        
        send_counts[owner_rank]++;
    }
    
    //we twll all the ranks how many  values we need
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);    
    

    int* send_displs = calloc(size, sizeof(int));
    int* recv_displs = calloc(size, sizeof(int));

    //we need to know from what position starts the block for the specific rank
    send_displs[0] = 0;
    recv_displs[0] = 0;
    for(i = 1; i < size; i++){
        send_displs[i] = send_displs[i-1] + send_counts[i-1];
        recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
    }

    //initialising buffers
    int total_send = send_displs[size-1] + send_counts[size-1];
    int total_recv = recv_displs[size-1] + recv_counts[size-1];

    
    int* indices_to_request = malloc(total_send * sizeof(int));
    int* indices_requested_from_me = malloc(total_recv * sizeof(int));
    

    int* current_pos = calloc(size, sizeof(int));
    for(i = 0; i < csr->ghost_count; i++){
        int global_col = csr->ghost_indices[i];
        int owner_rank = global_col % size;
        
        int pos = send_displs[owner_rank] + current_pos[owner_rank];
        indices_to_request[pos] = global_col;
        current_pos[owner_rank]++;
    }

    //we send the indices
    MPI_Alltoallv(indices_to_request, send_counts, send_displs, MPI_INT, indices_requested_from_me, recv_counts, recv_displs, MPI_INT, MPI_COMM_WORLD);
        
    //we prepare the values to send
    double* values_to_send = malloc(total_recv * sizeof(double));

    for(i = 0; i < total_recv; i++){
        int global_idx = indices_requested_from_me[i];
        int local_idx = global_idx / size;  
        values_to_send[i] = local_vector[local_idx];
    }
    
    //we exchange the values
    
    double* values_received = malloc(total_send * sizeof(double));
    
    MPI_Alltoallv(values_to_send, recv_counts, recv_displs, MPI_DOUBLE, values_received, send_counts, send_displs, MPI_DOUBLE, MPI_COMM_WORLD);
        
    //we copy the values in the final array (ghost_vector)
    memset(current_pos, 0, size * sizeof(int));
    for(i = 0; i < csr->ghost_count; i++){
        int global_col = csr->ghost_indices[i];
        int owner_rank = global_col % size;
        int pos = send_displs[owner_rank] + current_pos[owner_rank];
        (*ghost_vector_ptr)[i] = values_received[pos];
        current_pos[owner_rank]++;
    }
    
    free(send_counts);
    free(recv_counts);
    free(send_displs);
    free(recv_displs);
    free(indices_to_request);
    free(indices_requested_from_me);
    free(values_to_send);
    free(values_received);
    free(current_pos);
}



int compare_ints(const void *a, const void *b){
    return (*(int*)a - *(int*)b);
}


