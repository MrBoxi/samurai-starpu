#!/usr/bin/env python3
import os
import sys

# Ensure execution from xp directory so common.py does not fail
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != SCRIPT_DIR:
    os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from common import PERF_DIR, PLOTS_DIR

def generate_comparison_plots():
    csv_file = os.path.join(PERF_DIR, "perf_results.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Cannot plot.")
        return
        
    df = pd.read_csv(csv_file)
    
    # Map units to a unified column
    df['units'] = np.where(df['version'] == 'mpi', df['mpi_size'], df['num_threads'])
    
    # Compute stats per config (level, version, units, max_iter): mean, min, max
    agg_df = df.groupby(['level', 'version', 'units', 'max_iter'], as_index=False)['time'].agg(
        time_mean='mean', time_min='min', time_max='max'
    )
    
    # Apply a nice seaborn style
    sns.set_theme(style="whitegrid")
    
    # ------------------ Plot 1: Execution Time vs. Units ------------------
    levels = sorted(df['level'].unique())
    fig, axes = plt.subplots(1, len(levels), figsize=(18, 5), sharey=False)
    if len(levels) == 1:
        axes = [axes]
        
    colors = {"omp": "#1f77b4", "mpi": "#ff7f0e", "starpu": "#2ca02c"}
    markers = {"omp": "o", "mpi": "s", "starpu": "^"}
    labels = {"omp": "OpenMP", "mpi": "MPI", "starpu": "StarPU (Uniform)"}
    

    
    for i, lvl in enumerate(levels):
        ax = axes[i]
        
        # Get the latest max_iter for this level to avoid mixing old and new configurations
        lvl_df = df[df['level'] == lvl]
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        
        lvl_data = agg_df[(agg_df['level'] == lvl) & (agg_df['max_iter'] == latest_max_iter)]
        
        for version in ["omp", "mpi", "starpu"]:
            v_data = lvl_data[lvl_data['version'] == version].sort_values('units')
            if not v_data.empty:
                yerr_low = v_data['time_mean'] - v_data['time_min']
                yerr_high = v_data['time_max'] - v_data['time_mean']
                ax.errorbar(v_data['units'], v_data['time_mean'], 
                            yerr=[yerr_low, yerr_high],
                            marker=markers[version], color=colors[version], linewidth=2, markersize=8,
                            capsize=4, capthick=1.5,
                            label=labels[version])
                
        ax.set_title(f"Level {lvl} ($2^{{{lvl}}}\\times 2^{{{lvl}}}$ cells)\n({latest_max_iter} iterations)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Calculation Units (Threads / Processes)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Execution Time (seconds)", fontsize=12)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(lvl_data['units'].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=11)
        
    plt.suptitle("Advection 2D Performance Comparison: StarPU vs. OpenMP vs. MPI", fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()
    
    output_dir = os.path.join(PLOTS_DIR, "uniform")
    os.makedirs(output_dir, exist_ok=True)

    out_perf_base = os.path.join(output_dir, "performance_comparison")
    for ext in ["png", "pdf"]:
        plt.savefig(f"{out_perf_base}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved execution time plot to {out_perf_base}.png and {out_perf_base}.pdf")
    
    # ------------------ Plot 2: Speedup vs. Units ------------------
    speedup_rows = []
    for lvl in levels:
        lvl_df = df[df['level'] == lvl]
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        
        for version in ["omp", "mpi", "starpu"]:
            v_raw = lvl_df[(lvl_df['version'] == version) & (lvl_df['max_iter'] == latest_max_iter)]
            if v_raw.empty:
                continue
            # Find baseline mean time at units = 1
            t1_raw = v_raw[v_raw['units'] == 1]
            if t1_raw.empty:
                min_units = v_raw['units'].min()
                t1_mean = v_raw[v_raw['units'] == min_units]['time'].mean()
            else:
                t1_mean = t1_raw['time'].mean()
                
            for idx, row in v_raw.iterrows():
                speedup = t1_mean / row['time']
                speedup_rows.append({
                    'level': lvl,
                    'max_iter': latest_max_iter,
                    'version': version,
                    'units': row['units'],
                    'speedup': speedup
                })
                
    df_speedup = pd.DataFrame(speedup_rows)
    agg_speedup = df_speedup.groupby(['level', 'version', 'units', 'max_iter'], as_index=False)['speedup'].agg(
        speedup_mean='mean', speedup_min='min', speedup_max='max'
    )
    
    fig, axes = plt.subplots(1, len(levels), figsize=(18, 5), sharey=True)
    if len(levels) == 1:
        axes = [axes]
        
    for i, lvl in enumerate(levels):
        ax = axes[i]
        
        lvl_df = df[df['level'] == lvl]
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        
        lvl_data = agg_speedup[(agg_speedup['level'] == lvl) & (agg_speedup['max_iter'] == latest_max_iter)]
        
        # Plot ideal speedup line
        min_u = lvl_data['units'].min()
        max_u = lvl_data['units'].max()
        ax.plot([min_u, max_u], [min_u, max_u], linestyle='--', color='gray', label='Ideal Scaling', alpha=0.7)
        
        for version in ["omp", "mpi", "starpu"]:
            v_data = lvl_data[lvl_data['version'] == version].sort_values('units')
            if not v_data.empty:
                yerr_low = v_data['speedup_mean'] - v_data['speedup_min']
                yerr_high = v_data['speedup_max'] - v_data['speedup_mean']
                ax.errorbar(v_data['units'], v_data['speedup_mean'], 
                            yerr=[yerr_low, yerr_high],
                            marker=markers[version], color=colors[version], linewidth=2, markersize=8,
                            capsize=4, capthick=1.5,
                            label=labels[version])
                
        ax.set_title(f"Level {lvl} ($2^{{{lvl}}}\\times 2^{{{lvl}}}$ cells)\n({latest_max_iter} iterations)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Calculation Units (Threads / Processes)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Speedup (relative to 1 unit)", fontsize=12)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(lvl_data['units'].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(bottom=0, top=max_u + 2)
        ax.legend(fontsize=11)
        
    plt.suptitle("Advection 2D Speedup Comparison: StarPU vs. OpenMP vs. MPI", fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()
    
    out_speedup_base = os.path.join(output_dir, "speedup_comparison")
    for ext in ["png", "pdf"]:
        plt.savefig(f"{out_speedup_base}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved speedup plot to {out_speedup_base}.png and {out_speedup_base}.pdf")

if __name__ == "__main__":
    generate_comparison_plots()
