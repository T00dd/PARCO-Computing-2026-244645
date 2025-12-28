import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (16, 10)})

def load_and_process_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    
    df['matrix_name'] = df['matrix'].apply(lambda x: str(x).split('/')[-1].replace('.mtx', ''))
    
    cols = ['L1_dcache_loads', 'L1_dcache_load_misses', 'LLC_loads', 'LLC_misses']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['chunk'] = pd.to_numeric(df['chunk'], errors='coerce')

    df = df[(df['schedule'] == 'static') & (df['chunk'] == 100)]
    
    if df.empty:
        print("Warning: No data found for schedule='static' and chunk=100")
        return df

    grouped = df.groupby(['matrix_name', 'threads'])
    
    def get_fourth_run(g):
        if len(g) > 3:
            return g.iloc[3]
        return g.iloc[-1]
    
    stable_runs = grouped.apply(get_fourth_run).reset_index(drop=True)
    
    stable_runs['L1_miss_rate'] = (stable_runs['L1_dcache_load_misses'] / stable_runs['L1_dcache_loads']) * 100
    stable_runs['LLC_miss_rate'] = (stable_runs['LLC_misses'] / stable_runs['LLC_loads']) * 100
    
    return stable_runs

def plot_graphs(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    if df.empty:
        print("Empty dataset, cannot generate graphs.")
        return

    x_ticks = [1, 2, 4, 8, 16, 32, 64]
    if df['threads'].max() > 64:
        x_ticks.append(df['threads'].max())

    matrices = df['matrix_name'].unique()
    palette = dict(zip(matrices, sns.color_palette("deep", len(matrices))))

    def draw_plot(metric_col, ylabel, title, filename):
        plt.figure()
        
        ax = sns.lineplot(data=df, x='threads', y=metric_col, hue='matrix_name', marker='s', linewidth=4, markersize=14, palette=palette)
        
        ax.set_xscale('log', base=2)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xticks(x_ticks)
        
        plt.title(title, fontsize=32, fontweight='bold', pad=20)
        plt.ylabel(ylabel, fontsize=28, fontweight='bold')
        plt.xlabel('Threads', fontsize=28, fontweight='bold')
        
        ax.tick_params(axis='both', which='major', labelsize=24, width=2, length=8)
        
        plt.grid(True, which="major", linestyle="--", linewidth=1.5)
        
        plt.legend(title='Matrix', bbox_to_anchor=(1.02, 1), loc='upper left', 
                   fontsize=24, title_fontsize=28, frameon=True, framealpha=1, edgecolor='black')
        
        plt.tight_layout()
        
        outfile = os.path.join(output_dir, filename)
        plt.savefig(outfile, dpi=300) 
        plt.close()
        print(f"Saved: {outfile}")

    draw_plot('L1_miss_rate', 'L1 Miss Rate (%)', 
              'L1 Cache Miss Rate (Static, 100)', 
              "L1_miss_rate_static_100.png")

    draw_plot('LLC_miss_rate', 'LLC Miss Rate (%)', 
              'LLC Miss Rate (Static, 100)', 
              "LLC_miss_rate_static_100.png")

if __name__ == "__main__":
    base_path = Path(os.path.abspath(__file__)).parent
    input_file = (base_path / ".." / "results" / "perf_results.csv").resolve()
    output_dir = (base_path / ".." / "plots").resolve()
    
    try:
        print(f"Reading data from: {input_file}")
        data = load_and_process_data(str(input_file))
        
        print(f"Generating graphs in: {output_dir}")
        plot_graphs(data, str(output_dir))
        
        print("Done.")
        
    except Exception as e:
        print(f"Error: {e}")