import pandas as pd
import matplotlib.pyplot as plt
import os

input_file = '../results/time_results_weak.csv'
output_dir = '../plots/'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file, skipinitialspace=True)
df = df.sort_values('size')

t1 = df.iloc[0]['time_multiplication_ms'] + df.iloc[0]['time_comunication_ms']
df['weak_eff'] = t1 / (df['time_multiplication_ms'] + df['time_comunication_ms'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
x_labels = df['size'].astype(str)

ax1.bar(x_labels, df['time_multiplication_ms'], label='Multiplication (SpMV)', color='skyblue')
ax1.bar(x_labels, df['time_comunication_ms'], 
        bottom=df['time_multiplication_ms'], label='Communication (Ghost)', color='salmon')

ax1.set_xlabel('Number of processes (P)')
ax1.set_ylabel('Toltal times (ms)')
ax1.set_title('Weak Scaling: Execution time')
ax1.legend(loc='upper left')
ax1.grid(axis='y', linestyle='--', alpha=0.3)

ax2.plot(x_labels, df['weak_eff'], color='darkgreen', marker='D', linewidth=2, label='Efficiency')
ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Ideal')

ax2.set_xlabel('Num of processes (P)')
ax2.set_ylabel('Efficiency (T1 / Tp)')
ax2.set_title('Weak Scaling: Parallel efficiency')
ax2.set_ylim(0, max(1.2, df['weak_eff'].max() + 0.1))
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'weak_scaling_split.png'))
print(f"Generated plots in {output_dir}")