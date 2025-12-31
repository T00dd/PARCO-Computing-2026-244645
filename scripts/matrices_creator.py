
import numpy as np
from scipy.sparse import random as sparse_random
from scipy.io import mmwrite
import os

BASE_SIZE = 5000           
BASE_NNZ_PER_ROW = 50      
NUM_MATRICES = 8           
OUTPUT_DIR = "../data"   

num_processes = [1, 2, 4, 8, 16, 32, 64, 128]

def generate_sparse_matrix(n_rows, n_cols, nnz_per_row, seed=42):

    np.random.seed(seed)
    
    density = min(nnz_per_row / n_cols, 0.1) 
    
    matrix = sparse_random(n_rows, n_cols, density=density, 
                          format='coo', dtype=np.float64,
                          random_state=seed)
    
    matrix.data = np.random.uniform(-10, 10, size=matrix.data.shape)
    
    return matrix

def save_matrix_market(matrix, filename):
    mmwrite(filename, matrix, comment=f'Generated sparse matrix: {matrix.shape[0]}x{matrix.shape[1]}, nnz={matrix.nnz}')
    print(f"  saved: {filename}")
    print(f"  dimension: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"  non-zero: {matrix.nnz}")
    print(f"  density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.4f}%")
    print()

print("="*70)
print("GENERATING MATRICES FOR WEAK SCALING")
print("="*70)
print(f"Base dimention: {BASE_SIZE}x{BASE_SIZE}")
print(f"NNZ for row: {BASE_NNZ_PER_ROW}")
print(f"Number of matrices: {NUM_MATRICES}")
print("="*70)
print()

# Genera le matrici
for i, num_proc in enumerate(num_processes, 1):
    size = int(BASE_SIZE * np.sqrt(num_proc))
    
    print(f"[{i}/{NUM_MATRICES}] Generating matrices for {num_proc} processes...")
    
    matrix = generate_sparse_matrix(size, size, BASE_NNZ_PER_ROW, seed=42+i)
    
    filename = os.path.join(OUTPUT_DIR, f"matrix_weak_{num_proc}proc_{size}x{size}.mtx")
    
    save_matrix_market(matrix, filename)

print("==============================================================")
print("GENERATION FINISHED!")
print("==============================================================")