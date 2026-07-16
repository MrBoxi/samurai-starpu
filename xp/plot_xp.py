#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from common import ROOT_PROJECT_PATH, PERF_DIR

def generate_comparison_plots():
    csv_file = os.path.join(PERF_DIR, "perf_results.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Cannot plot.")
        return
        
    df = pd.read_csv(csv_file)
    
    # Map units to a unified column
    df['units'] = np.where(df['version'] == 'mpi', df['mpi_size'], df['num_threads'])
    
    # Compute average execution time per config (level, version, units, max_iter)
    grouped = df.groupby(['level', 'version', 'units', 'max_iter'], as_index=False)['time'].mean()
    
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
    
    # We will store the selected lvl_data for speedup calculation
    latest_grouped_rows = []
    
    for i, lvl in enumerate(levels):
        ax = axes[i]
        
        # Get the latest max_iter for this level to avoid mixing old and new configurations
        lvl_df = df[df['level'] == lvl]
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        
        lvl_data = grouped[(grouped['level'] == lvl) & (grouped['max_iter'] == latest_max_iter)]
        latest_grouped_rows.append(lvl_data)
        
        for version in ["omp", "mpi", "starpu"]:
            v_data = lvl_data[lvl_data['version'] == version].sort_values('units')
            if not v_data.empty:
                ax.plot(v_data['units'], v_data['time'], 
                        marker=markers[version], color=colors[version], linewidth=2, markersize=8,
                        label=labels[version])
                
        ax.set_title(f"Level {lvl} ($2^{{{lvl}}}\\times 2^{{{lvl}}}$ cells)\n({latest_max_iter} iterations)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Calculation Units (Threads / Processes)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Execution Time (seconds)", fontsize=12)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(lvl_data['units'].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend(fontsize=11)
        
    plt.suptitle("Advection 2D Performance Comparison: StarPU vs. OpenMP vs. MPI", fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()
    
    out_perf = os.path.join(ROOT_PROJECT_PATH, "performance_comparison.png")
    plt.savefig(out_perf, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved execution time plot to {out_perf}")
    
    # ------------------ Plot 2: Speedup vs. Units ------------------
    speedup_rows = []
    for lvl_data in latest_grouped_rows:
        if lvl_data.empty:
            continue
        lvl = lvl_data['level'].iloc[0]
        latest_max_iter = lvl_data['max_iter'].iloc[0]
        
        for version in ["omp", "mpi", "starpu"]:
            v_data = lvl_data[lvl_data['version'] == version]
            if v_data.empty:
                continue
            # Find baseline time at units = 1
            t1_row = v_data[v_data['units'] == 1]
            if t1_row.empty:
                min_units = v_data['units'].min()
                t1 = v_data[v_data['units'] == min_units]['time'].values[0]
            else:
                t1 = t1_row['time'].values[0]
                
            for idx, row in v_data.iterrows():
                speedup = t1 / row['time']
                speedup_rows.append({
                    'level': lvl,
                    'max_iter': latest_max_iter,
                    'version': version,
                    'units': row['units'],
                    'speedup': speedup
                })
                
    df_speedup = pd.DataFrame(speedup_rows)
    
    fig, axes = plt.subplots(1, len(levels), figsize=(18, 5), sharey=True)
    if len(levels) == 1:
        axes = [axes]
        
    for i, lvl in enumerate(levels):
        ax = axes[i]
        
        # Get the latest max_iter for this level
        lvl_df = df[df['level'] == lvl]
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        
        lvl_data = df_speedup[(df_speedup['level'] == lvl) & (df_speedup['max_iter'] == latest_max_iter)]
        
        # Plot ideal speedup line
        min_u = lvl_data['units'].min()
        max_u = lvl_data['units'].max()
        ax.plot([min_u, max_u], [min_u, max_u], linestyle='--', color='gray', label='Ideal Scaling', alpha=0.7)
        
        for version in ["omp", "mpi", "starpu"]:
            v_data = lvl_data[lvl_data['version'] == version].sort_values('units')
            if not v_data.empty:
                ax.plot(v_data['units'], v_data['speedup'], 
                        marker=markers[version], color=colors[version], linewidth=2, markersize=8,
                        label=labels[version])
                
        ax.set_title(f"Level {lvl} ($2^{{{lvl}}}\\times 2^{{{lvl}}}$ cells)\n({latest_max_iter} iterations)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Calculation Units (Threads / Processes)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Speedup (relative to 1 unit)", fontsize=12)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(lvl_data['units'].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(0, max_u + 2)
        ax.legend(fontsize=11)
        
    plt.suptitle("Advection 2D Speedup Comparison: StarPU vs. OpenMP vs. MPI", fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()
    
    out_speedup = os.path.join(ROOT_PROJECT_PATH, "speedup_comparison.png")
    plt.savefig(out_speedup, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved speedup plot to {out_speedup}")

if __name__ == "__main__":
    generate_comparison_plots()
