import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmwrite
import os


BASE_ROWS = 256
BASE_NNZ_PER_ROW = 50
NUM_MATRICES = 8
OUTPUT_DIR = "../data"


num_processes = [1, 2, 4, 8, 16, 32, 64, 128]


os.makedirs(OUTPUT_DIR, exist_ok=True)



def generate_sparse_matrix_fixed_nnz(n_rows, n_cols, nnz_per_row, seed=42):
    """
    Genera una matrice sparsa con NNZ per riga approssimativamente costante.
    Gli elementi sono ordinati per COLONNE (column-major).
    """
    np.random.seed(seed)
    
    variation = int(nnz_per_row * 0.2)
    
    row_indices = []
    col_indices = []
    values = []
    
    for row in range(n_rows):
        actual_nnz = nnz_per_row + np.random.randint(-variation, variation + 1)
        actual_nnz = max(1, min(actual_nnz, n_cols))
        
        cols = np.random.choice(n_cols, size=actual_nnz, replace=False)
        cols.sort()
        
        vals = np.random.uniform(-10, 10, size=actual_nnz)
        
        row_indices.extend([row] * actual_nnz)
        col_indices.extend(cols)
        values.extend(vals)
    
    row_indices = np.array(row_indices, dtype=np.int32)
    col_indices = np.array(col_indices, dtype=np.int32)
    values = np.array(values, dtype=np.float64)
    

    sorted_indices = np.lexsort((row_indices, col_indices))
    row_indices = row_indices[sorted_indices]
    col_indices = col_indices[sorted_indices]
    values = values[sorted_indices]
    

    matrix = coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_rows, n_cols),
        dtype=np.float64
    )
    
    return matrix



def save_matrix_market(matrix, filename):
    """
    Salva la matrice in formato MatrixMarket
    """
    mmwrite(filename, matrix, 
            comment=f'Weak scaling matrix: {matrix.shape[0]}x{matrix.shape[1]}, nnz={matrix.nnz}',
            field='real',
            precision=15,
            symmetry='general')
    
    avg_nnz_per_row = matrix.nnz / matrix.shape[0]
    print(f"  Saved: {filename}")
    print(f"  Dimension: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"  Non-zero: {matrix.nnz}")
    print(f"  Avg NNZ/row: {avg_nnz_per_row:.2f}")
    print(f"  Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.4f}%")
    print()


print("=" * 70)
print("GENERATING MATRICES FOR WEAK SCALING (COLUMN-MAJOR ORDER)")
print("=" * 70)
print(f"Base rows per process: {BASE_ROWS}")
print(f"Target NNZ per row: {BASE_NNZ_PER_ROW} (±20%)")
print(f"Number of matrices: {NUM_MATRICES}")
print(f"Ordering: COLUMN-MAJOR (sorted by columns first)")
print("=" * 70)
print()


for i, num_proc in enumerate(num_processes, 1):
    n_rows = BASE_ROWS * num_proc
    n_cols = n_rows
    
    print(f"[{i}/{NUM_MATRICES}] Generating matrix for {num_proc} processes...")
    
    matrix = generate_sparse_matrix_fixed_nnz(n_rows, n_cols, BASE_NNZ_PER_ROW, seed=42 + i)
    
    filename = os.path.join(OUTPUT_DIR, f"matrix_weak_{num_proc}.mtx")
    
    save_matrix_market(matrix, filename)


print("=" * 70)
print("WEAK SCALING MATRICES GENERATED CORRECTLY!")
print("=" * 70)
print(f"\nExpected scaling:")
for p in num_processes:
    expected_nnz = BASE_ROWS * p * BASE_NNZ_PER_ROW
    print(f"  P={p:3d}: {BASE_ROWS * p:6d} rows, ~{expected_nnz:8d} NNZ")

print("\nAll matrices sorted in COLUMN-MAJOR order!")
