import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

INPUT_PATH = '../results/time_results_strong.csv'
OUTPUT_DIR = '../plots/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(INPUT_PATH)
except FileNotFoundError:
    print(f"Error: file not found in {INPUT_PATH}")
    exit(1)

df['total_time'] = df['time_comunication_ms'] + df['time_multiplication_ms']
matrices = df['matrix'].unique()
n_matrices = len(matrices)

custom_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

# ============ SPEEDUP & EFFICIENCY ============
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

max_speedup_real = 0
for matrix in matrices:
    m_data = df[df['matrix'] == matrix].sort_values('size')
    t1 = m_data[m_data['size'] == 1]['total_time'].values[0]
    m_data['speedup'] = t1 / m_data['total_time']
    max_speedup_real = max(max_speedup_real, m_data['speedup'].max())

y_limit = min(df['size'].max(), max_speedup_real * 1.3) 

for i, matrix in enumerate(matrices):
    m_name = matrix.split('/')[-1]
    m_data = df[df['matrix'] == matrix].sort_values('size')
    
    t1 = m_data[m_data['size'] == 1]['total_time'].values[0]
    m_data['speedup'] = t1 / m_data['total_time']
    m_data['efficiency'] = m_data['speedup'] / m_data['size']
    
    color = custom_colors[i % len(custom_colors)]
    ax1.plot(m_data['size'], m_data['speedup'], marker='o', markersize=8, 
             label=m_name, color=color, linewidth=2.5)
    ax2.plot(m_data['size'], m_data['efficiency'], marker='s', markersize=8, 
             label=m_name, color=color, linewidth=2.5)

ax1.plot([1, y_limit], [1, y_limit], 
         linestyle='--', color='black', alpha=0.8, linewidth=2.5, 
         label='Ideale', zorder=10)

ax1.set_title('Strong Scaling: speedup', fontsize=18, fontweight='bold')
ax1.set_xlabel('Processes MPI', fontsize=14)
ax1.set_ylabel('Speedup', fontsize=14)
ax1.set_ylim(0, y_limit) 
ax1.legend(fontsize=10, loc='upper left', ncol=2)
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.axhline(y=1, linestyle='--', color='black', alpha=0.8, linewidth=2.5, 
            label='Ideale', zorder=10)
ax2.set_title('Strong scaling: efficiency', fontsize=18, fontweight='bold')
ax2.set_xlabel('Processes MPI', fontsize=14)
ax2.set_ylabel('Efficiency', fontsize=14)
ax2.set_ylim(0, 1.1)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'consolidated_scaling.png'), dpi=300)

# ============ LOAD BALANCE & COMM VOLUME - LAYOUT ORIZZONTALE ============
# RIMUOVI size=1 perché non ha imbalance
df_multi_proc = df[df['size'] > 1].copy()

# Layout orizzontale: 2 righe × n_matrices colonne
fig2, axes = plt.subplots(2, n_matrices, figsize=(3.2 * n_matrices, 6.5), sharex=True)

for i, matrix in enumerate(matrices):
    m_name = matrix.split('/')[-1]
    # Filtra solo size > 1
    m_data = df_multi_proc[df_multi_proc['matrix'] == matrix].sort_values('size').copy()
    
    if len(m_data) == 0:
        continue
    
    ranks_str = m_data['size'].astype(str)
    
    # ===== Load Imbalance in % (riga 0) =====
    m_data['nnz_imbalance_pct'] = ((m_data['global_max_nnz'] - m_data['global_min_nnz']) / m_data['avg_nnz']) * 100
    
    axes[0, i].bar(ranks_str, m_data['nnz_imbalance_pct'], color='#a6cee3', 
                   edgecolor='navy', alpha=0.8, linewidth=1.2)
    axes[0, i].set_title(f'{m_name}', fontsize=13, fontweight='bold', pad=8)
    axes[0, i].axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=2, label='10% threshold')
    axes[0, i].tick_params(axis='both', labelsize=10, width=1.2)
    axes[0, i].grid(True, linestyle=':', alpha=0.4)
    
    # Label Y solo per la prima colonna
    if i == 0:
        axes[0, i].set_ylabel('Load Imbalance (%)', fontsize=12, fontweight='bold')
        axes[0, i].legend(fontsize=10, loc='upper left')
    
    # ===== Comm Imbalance in % (riga 1) =====
    m_data['comm_imbalance_pct'] = np.where(
        m_data['avg_comm'] > 0,
        ((m_data['global_max_comm'] - m_data['global_min_comm']) / m_data['avg_comm']) * 100,
        0
    )
    
    axes[1, i].bar(ranks_str, m_data['comm_imbalance_pct'], color='#fdbf6f', 
                   edgecolor='darkorange', alpha=0.8, linewidth=1.2)
    axes[1, i].axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=2, label='10% threshold')
    axes[1, i].set_xlabel('Processes MPI', fontsize=11, fontweight='bold')
    axes[1, i].tick_params(axis='both', labelsize=10, width=1.2)
    axes[1, i].grid(True, linestyle=':', alpha=0.4)
    
    # Label Y solo per la prima colonna
    if i == 0:
        axes[1, i].set_ylabel('Comm Imbalance (%)', fontsize=12, fontweight='bold')
        axes[1, i].legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.28)
plt.savefig(os.path.join(OUTPUT_DIR, 'bonus_imbalance_pct.png'), dpi=300)

print("Generated plots in ../plots/")

# ============ BEST PERFORMANCE SUMMARY TABLE ============
print("\n" + "="*130)
print("BEST SPEEDUP SUMMARY")
print("="*130)
print(f"{'Matrix':<20} {'Best Speedup':>12} {'Processes':>10} {'GFLOPS':>10} {'Total (ms)':>12} {'Comm (ms)':>12} {'Calc (ms)':>12}")
print("-"*130)

best_results = []

for matrix in matrices:
    m_name = matrix.split('/')[-1]
    m_data = df[df['matrix'] == matrix].sort_values('size').copy()
    
    # Calculate speedup
    t1 = m_data[m_data['size'] == 1]['total_time'].values[0]
    m_data['speedup'] = t1 / m_data['total_time']
    
    # Find best speedup
    best_idx = m_data['speedup'].idxmax()
    best_row = m_data.loc[best_idx]
    
    best_speedup = best_row['speedup']
    best_procs = best_row['size']
    best_gflops = best_row['gflops']
    best_time = best_row['total_time']
    best_comm = best_row['time_comunication_ms']
    best_calc = best_row['time_multiplication_ms']
    
    print(f"{m_name:<20} {best_speedup:>12.2f} {best_procs:>10.0f} {best_gflops:>10.2f} {best_time:>12.2f} {best_comm:>12.4f} {best_calc:>12.4f}")
    
    best_results.append({
        'matrix': m_name,
        'best_speedup': best_speedup,
        'processes': best_procs,
        'gflops': best_gflops,
        'time_total_ms': best_time,
        'time_comm_ms': best_comm,
        'time_calc_ms': best_calc
    })

print("="*130)

# Save to CSV
best_df = pd.DataFrame(best_results)
best_df.to_csv(os.path.join(OUTPUT_DIR, 'best_speedup_summary.csv'), index=False)
print(f"\nBest speedup summary saved to {os.path.join(OUTPUT_DIR, 'best_speedup_summary.csv')}")
