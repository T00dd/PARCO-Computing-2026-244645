import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (20, 10)})

def load_time_results(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    df["matrix_name"] = df["matrix"].apply(lambda x: os.path.basename(str(x)).replace(".mtx", ""))
    df["chunk size"] = df["chunk size"].fillna(0).astype(int)
    return df

def compute_percentile(df):
    agg = df.groupby(["matrix_name","threads","schedules type","chunk size"])["time (ms)"] \
            .quantile(0.9).reset_index()

    seq = agg[agg["schedules type"]=="seq"][["matrix_name","time (ms)"]] \
           .rename(columns={"time (ms)":"seq_time"})

    merged = pd.merge(agg, seq, on="matrix_name")
    return merged

def print_best_speedup_table(df):
    matrices = df["matrix_name"].unique()
    
    print("\n" + "="*95)
    print(f"{'MATRIX':<20} | {'SPEEDUP':<10} | {'THREADS':<8} | {'SCHEDULE':<10} | {'CHUNK':<6} | {'TIME (ms)':<10}")
    print("="*95)

    for matrix in matrices:
        subset = df[df["matrix_name"] == matrix]
        
        if subset.empty: continue
        seq_time = subset.iloc[0]["seq_time"]
        
        if pd.isna(seq_time):
            print(f"{matrix:<20} | N/A (No seq)")
            continue

        parallel = subset[subset["schedules type"] != "seq"]
        
        if parallel.empty:
            print(f"{matrix:<20} | N/A (No par)")
            continue

        best_idx = parallel["time (ms)"].idxmin()
        best_row = parallel.loc[best_idx]
        
        best_time = best_row["time (ms)"]
        speedup = seq_time / best_time
        
        th = best_row["threads"]
        sch = best_row["schedules type"]
        chk = best_row["chunk size"]

        print(f"{matrix:<20} | "
              f"{speedup:>8.2f}x | "
              f"{th:>8} | "
              f"{sch:<10} | "
              f"{chk:<6} | "
              f"{best_time:>10.5f}")

    print("="*95 + "\n")

def plot_combined_execution_time(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    matrices = df["matrix_name"].unique()
    n_matrices = len(matrices)
    
    fig, axes = plt.subplots(2, 3, constrained_layout=True)
    axes_flat = axes.flatten()

    for k in range(n_matrices, len(axes_flat)):
        fig.delaxes(axes_flat[k])

    for i, matrix in enumerate(matrices):
        ax = axes_flat[i]
        subset = df[df["matrix_name"]==matrix]
        parallel = subset[subset["schedules type"]!="seq"]
        
        seq_time_val = subset.iloc[0]["seq_time"]

        if parallel.empty:
            ax.text(0.5, 0.5, "No Parallel Data", ha='center', va='center')
            continue

        best_idx = parallel.groupby(["threads","schedules type"])["time (ms)"].idxmin()
        best_df = parallel.loc[best_idx].sort_values("threads")
        
        schedules = best_df["schedules type"].unique()
        
        color_map = {
            'static': '#1f77b4',  # Blue
            'dynamic': '#ff7f0e', # Orange
            'guided': '#2ca02c'   # Green
        }

        for sched in schedules:
            sched_data = best_df[best_df["schedules type"] == sched]
            
            min_row = sched_data.loc[sched_data["time (ms)"].idxmin()]
            best_chunk_label = int(min_row["chunk size"])
            
            label_str = f"{sched} (chunk: {best_chunk_label})"
            
            ax.plot(sched_data["threads"], sched_data["time (ms)"], 
                    marker='o', markersize=6, linewidth=2,
                    label=label_str, color=color_map.get(sched, 'black'))

        if not np.isnan(seq_time_val):
            ax.axhline(y=seq_time_val, color='red', linestyle='--', linewidth=1.5, 
                       label=f'Sequential ({seq_time_val:.5f} ms)')

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.minorticks_on()
        ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.8, alpha=0.8)

        y_max = max(seq_time_val, parallel["time (ms)"].max()) if not np.isnan(seq_time_val) else parallel["time (ms)"].max()
        ax.set_ylim(bottom=0, top=y_max * 1.05)
        
        ax.set_title(matrix, fontsize=14, fontweight='bold')
        ax.set_xlabel("Threads", fontsize=11)
        ax.set_ylabel("Time (ms)", fontsize=11)
        
        ax.legend(fontsize=13, loc='best', framealpha=0.9, edgecolor='gray')

    fig.suptitle("Execution Time vs Threads Analysis (90th Percentile)", fontsize=20, fontweight='bold')
    
    output_filename = os.path.join(outdir, "Combined_Execution_Time_Final.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Combined plot saved in: {output_filename}")

if __name__ == "__main__":
    base_path = Path(os.path.abspath(__file__)).parent
    input_file_path = (base_path / ".." / "results" / "time_results.csv").resolve()
    output_dir_path = (base_path / ".." / "plots").resolve()

    try:
        print(f"Reading from: {input_file_path}")
        if not input_file_path.exists():
             raise FileNotFoundError(f"File not found: {input_file_path}")

        raw_data = load_time_results(str(input_file_path))
        
        print("Computing stats (90th percentile)...")
        stats_data = compute_percentile(raw_data)
        
        print_best_speedup_table(stats_data)
        
        print(f"Generating combined plot in: {output_dir_path}")
        plot_combined_execution_time(stats_data, str(output_dir_path))
        
        print("Done.")
        
    except Exception as e:
        print(f"Error: {e}")