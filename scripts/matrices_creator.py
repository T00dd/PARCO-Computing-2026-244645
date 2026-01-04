import numpy as np
from scipy.sparse import random as sparse_random
from scipy.io import mmwrite
import os

BASE_ROWS = 256
BASE_NNZ_PER_ROW = 50
NUM_MATRICES = 8
OUTPUT_DIR = "../data"

num_processes = [1, 2, 4, 8, 16, 32, 64, 128]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_sparse_matrix(n_rows, n_cols, nnz_per_row, seed=42):
    np.random.seed(seed)
    density = min(nnz_per_row / n_cols, 0.1)
    matrix = sparse_random(n_rows, n_cols, density=density, format='coo', dtype=np.float64, random_state=seed)
    matrix.data = np.random.uniform(-10, 10, size=matrix.data.shape)
    
    sort_idx = np.lexsort((matrix.row, matrix.col))
    matrix.row = matrix.row[sort_idx]
    matrix.col = matrix.col[sort_idx]
    matrix.data = matrix.data[sort_idx]
    
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
print(f"Base rows: {BASE_ROWS}")
print(f"NNZ per row: {BASE_NNZ_PER_ROW}")
print(f"Number of matrices: {NUM_MATRICES}")
print("="*70)
print()

for i, num_proc in enumerate(num_processes, 1):
    n_rows = BASE_ROWS * num_proc
    n_cols = n_rows
    
    print(f"[{i}/{NUM_MATRICES}] Generating matrix for {num_proc} processes...")
    
    matrix = generate_sparse_matrix(n_rows, n_cols, BASE_NNZ_PER_ROW, seed=42+i)
    
    filename = os.path.join(OUTPUT_DIR, f"matrix_weak_{num_proc}.mtx")
    
    save_matrix_market(matrix, filename)

print("="*70)
print("GENERATION FINISHED!")
print("="*70)
