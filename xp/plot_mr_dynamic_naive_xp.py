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
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import numpy as np

from common import PERF_DIR, PLOTS_DIR

try:
    from run_mr_dynamic_naive_xp import TASKS_PER_UNIT
except ImportError:
    TASKS_PER_UNIT = 1

def generate_comparison_plots():
    csv_file = os.path.join(PERF_DIR, "perf_results_mr_dynamic_naive.csv")
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Cannot plot.")
        return
        
    df = pd.read_csv(csv_file)
    if df.empty:
        print(f"Error: {csv_file} is empty. Cannot plot.")
        return
    
    # Normalize version strings
    df['version'] = df['version'].replace({'starpu_dynamic_naive': 'starpu'})
    
    # Map units to a unified column
    df['units'] = np.where(df['version'] == 'mpi', df['mpi_size'], df['num_threads'])
    
    # Ensure adapt_time column exists
    if 'adapt_time' not in df.columns:
        print("Warning: 'adapt_time' column missing in CSV. Filling with 0.0.")
        df['adapt_time'] = 0.0
    
    # Compute computation-only time (excluding adaptation)
    df['comp_time'] = (df['time'] - df['adapt_time']).clip(lower=0.0)

    # Detect tasks per unit from StarPU data if available
    sp_df = df[df['version'] == 'starpu']
    if not sp_df.empty and 'nb_tasks' in sp_df.columns:
        ratio = (sp_df['nb_tasks'] / sp_df['units']).median()
        tpu = int(round(ratio)) if not np.isnan(ratio) else TASKS_PER_UNIT
    else:
        tpu = TASKS_PER_UNIT

    # Filter to only keep rows matching the detected/configured tasks per unit for StarPU
    if 'nb_tasks' in df.columns:
        df = df[(df['version'] != 'starpu') | (df['nb_tasks'] == df['units'] * tpu)]

    # Compute stats per config (level, version, units, max_iter): mean, min, max
    agg_df = df.groupby(['level', 'version', 'units', 'max_iter'], as_index=False).agg(
        time_mean=('time', 'mean'), time_min=('time', 'min'), time_max=('time', 'max'),
        adapt_time_mean=('adapt_time', 'mean'), adapt_time_min=('adapt_time', 'min'), adapt_time_max=('adapt_time', 'max'),
        comp_time_mean=('comp_time', 'mean'), comp_time_min=('comp_time', 'min'), comp_time_max=('comp_time', 'max')
    )
    
    # Apply seaborn theme
    sns.set_theme(style="whitegrid")
    
    levels = sorted(df['level'].unique())
    colors = {"seq": "#9467bd", "omp": "#1f77b4", "mpi": "#ff7f0e", "starpu": "#2ca02c"}
    colors_adapt = {"seq": "#c5b0d5", "omp": "#aec7e8", "mpi": "#ffbb78", "starpu": "#98df8a"}
    markers = {"seq": "D", "omp": "o", "mpi": "s", "starpu": "^"}
    labels = {
        "seq": "Sequential",
        "omp": "OpenMP",
        "mpi": "MPI",
        "starpu": f"StarPU (Dynamic Naive, {tpu} task/unit)" if tpu == 1 else f"StarPU (Dynamic Naive, {tpu} tasks/unit)"
    }
    cell_counts = {11: 102616, 12: 202920, 13: 403096, 14: 805256, 15: 1608072}
    
    version_order = ["seq", "omp", "mpi", "starpu"]
    available_versions = [v for v in version_order if v in df['version'].values]

    # =========================================================================
    # Plot 1: Total Execution Time vs. Units
    # =========================================================================
    fig, axes = plt.subplots(1, len(levels), figsize=(6 * max(1, len(levels)), 5), sharey=False)
    if len(levels) == 1:
        axes = [axes]

    for i, lvl in enumerate(levels):
        ax = axes[i]
        lvl_df = df[df['level'] == lvl]
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        lvl_data = agg_df[(agg_df['level'] == lvl) & (agg_df['max_iter'] == latest_max_iter)]

        for version in available_versions:
            v_data = lvl_data[lvl_data['version'] == version].sort_values('units')
            if not v_data.empty:
                yerr_low = (v_data['time_mean'] - v_data['time_min']).to_numpy()
                yerr_high = (v_data['time_max'] - v_data['time_mean']).to_numpy()
                ax.errorbar(v_data['units'].to_numpy(), v_data['time_mean'].to_numpy(), 
                            yerr=[yerr_low, yerr_high],
                            marker=markers[version], color=colors[version], linewidth=2.2, markersize=8,
                            capsize=4, capthick=1.5,
                            label=labels[version])
                
        cells_str = f" (~{cell_counts[lvl]:,} cells)".replace(",", " ") if lvl in cell_counts else ""
        ax.set_title(f"Dynamic AMR: Level {lvl}{cells_str}\n({latest_max_iter} iterations)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Calculation Units (Threads / Processes)", fontsize=11, fontweight='semibold')
        if i == 0:
            ax.set_ylabel("Total Execution Time (seconds)", fontsize=11, fontweight='semibold')
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(lvl_data['units'].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)
        
    plt.suptitle("Advection 2D Dynamic AMR: Total Execution Time Comparison", fontsize=15, y=1.02, fontweight='bold')
    plt.tight_layout()
    output_dir = os.path.join(PLOTS_DIR, "mr_dynamic_naive")
    os.makedirs(output_dir, exist_ok=True)

    out_perf_base = os.path.join(output_dir, "performance_comparison_mr_dynamic_naive")
    for ext in ["png", "pdf"]:
        plt.savefig(f"{out_perf_base}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved total execution time plot to {out_perf_base}.png and {out_perf_base}.pdf")

    # =========================================================================
    # Plot 2: Total Speedup vs. Units
    # =========================================================================
    speedup_rows = []
    for lvl in levels:
        lvl_df = df[df['level'] == lvl]
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        
        for version in available_versions:
            v_raw = lvl_df[(lvl_df['version'] == version) & (lvl_df['max_iter'] == latest_max_iter)]
            if v_raw.empty:
                continue
            t1_raw = v_raw[v_raw['units'] == 1]
            t1_mean = t1_raw['time'].mean() if not t1_raw.empty else v_raw[v_raw['units'] == v_raw['units'].min()]['time'].mean()
                
            for _, row in v_raw.iterrows():
                speedup_rows.append({
                    'level': lvl,
                    'max_iter': latest_max_iter,
                    'version': version,
                    'units': row['units'],
                    'speedup': t1_mean / row['time'] if row['time'] > 0 else 0
                })
                
    df_speedup = pd.DataFrame(speedup_rows)
    agg_speedup = df_speedup.groupby(['level', 'version', 'units', 'max_iter'], as_index=False)['speedup'].agg(
        speedup_mean='mean', speedup_min='min', speedup_max='max'
    )
    
    fig, axes = plt.subplots(1, len(levels), figsize=(6 * max(1, len(levels)), 5), sharey=True)
    if len(levels) == 1:
        axes = [axes]
        
    max_total_sp = agg_speedup['speedup_max'].max() if not agg_speedup.empty else 1.0
    sp_top_lim = max(2.0, float(np.ceil(max_total_sp * 1.25 * 2.0) / 2.0))  # e.g., 2.0, 2.5

    for i, lvl in enumerate(levels):
        ax = axes[i]
        lvl_df = df[df['level'] == lvl]
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        lvl_data = agg_speedup[(agg_speedup['level'] == lvl) & (agg_speedup['max_iter'] == latest_max_iter)]
        
        min_u = lvl_data['units'].min()
        max_u = lvl_data['units'].max()
        # Ideal scaling line
        ax.plot([min_u, max_u], [min_u, max_u], linestyle='--', color='gray', label='Ideal Scaling', alpha=0.7)
        # Baseline reference line at y = 1.0
        ax.axhline(1.0, color='black', linestyle=':', linewidth=1.2, alpha=0.55, label='Baseline (1 unit)')
        
        for version in available_versions:
            v_data = lvl_data[lvl_data['version'] == version].sort_values('units')
            if not v_data.empty:
                yerr_low = (v_data['speedup_mean'] - v_data['speedup_min']).to_numpy()
                yerr_high = (v_data['speedup_max'] - v_data['speedup_mean']).to_numpy()
                ax.errorbar(v_data['units'].to_numpy(), v_data['speedup_mean'].to_numpy(), 
                            yerr=[yerr_low, yerr_high],
                            marker=markers[version], color=colors[version], linewidth=2.2, markersize=8,
                            capsize=4, capthick=1.5,
                            label=labels[version])
                
        cells_str = f" (~{cell_counts[lvl]:,} cells)".replace(",", " ") if lvl in cell_counts else ""
        ax.set_title(f"Dynamic AMR: Level {lvl}{cells_str}\n({latest_max_iter} iterations)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Calculation Units (Threads / Processes)", fontsize=11, fontweight='semibold')
        if i == 0:
            ax.set_ylabel("Total Speedup (relative to 1 unit)", fontsize=11, fontweight='semibold')
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(lvl_data['units'].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(bottom=0, top=sp_top_lim)
        ax.legend(fontsize=9, loc='upper left')
        
    plt.suptitle("Advection 2D Dynamic AMR: Total Speedup Comparison", fontsize=15, y=1.02, fontweight='bold')
    plt.tight_layout()
    out_speedup_base = os.path.join(output_dir, "speedup_comparison_mr_dynamic_naive")
    for ext in ["png", "pdf"]:
        plt.savefig(f"{out_speedup_base}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved total speedup plot to {out_speedup_base}.png and {out_speedup_base}.pdf")

    # =========================================================================
    # Plot 3: Computation-Only Time vs. Units (Excluding Adaptation)
    # =========================================================================
    fig, axes = plt.subplots(1, len(levels), figsize=(6 * max(1, len(levels)), 5), sharey=False)
    if len(levels) == 1:
        axes = [axes]

    for i, lvl in enumerate(levels):
        ax = axes[i]
        lvl_df = df[df['level'] == lvl]
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        lvl_data = agg_df[(agg_df['level'] == lvl) & (agg_df['max_iter'] == latest_max_iter)]

        for version in available_versions:
            v_data = lvl_data[lvl_data['version'] == version].sort_values('units')
            if not v_data.empty:
                yerr_low = (v_data['comp_time_mean'] - v_data['comp_time_min']).to_numpy()
                yerr_high = (v_data['comp_time_max'] - v_data['comp_time_mean']).to_numpy()
                ax.errorbar(v_data['units'].to_numpy(), v_data['comp_time_mean'].to_numpy(), 
                            yerr=[yerr_low, yerr_high],
                            marker=markers[version], color=colors[version], linewidth=2.2, markersize=8,
                            capsize=4, capthick=1.5,
                            label=labels[version])
                
        cells_str = f" (~{cell_counts[lvl]:,} cells)".replace(",", " ") if lvl in cell_counts else ""
        ax.set_title(f"Dynamic AMR: Level {lvl}{cells_str}\n({latest_max_iter} iterations)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Calculation Units (Threads / Processes)", fontsize=11, fontweight='semibold')
        if i == 0:
            ax.set_ylabel("Computation Time (seconds, excl. AMR)", fontsize=11, fontweight='semibold')
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(lvl_data['units'].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)
        
    plt.suptitle("Advection 2D Dynamic AMR: Pure Computation Time (Excl. Adaptation)", fontsize=15, y=1.02, fontweight='bold')
    plt.tight_layout()
    out_comp_time_base = os.path.join(output_dir, "computation_time_comparison_mr_dynamic_naive")
    for ext in ["png", "pdf"]:
        plt.savefig(f"{out_comp_time_base}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved computation time plot to {out_comp_time_base}.png and {out_comp_time_base}.pdf")

    # =========================================================================
    # Plot 4: Computation-Only Speedup vs. Units
    # =========================================================================
    comp_speedup_rows = []
    for lvl in levels:
        lvl_df = df[df['level'] == lvl]
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        
        for version in available_versions:
            v_raw = lvl_df[(lvl_df['version'] == version) & (lvl_df['max_iter'] == latest_max_iter)]
            if v_raw.empty:
                continue
            c1_raw = v_raw[v_raw['units'] == 1]
            c1_mean = c1_raw['comp_time'].mean() if not c1_raw.empty else v_raw[v_raw['units'] == v_raw['units'].min()]['comp_time'].mean()
                
            for _, row in v_raw.iterrows():
                comp_speedup_rows.append({
                    'level': lvl,
                    'max_iter': latest_max_iter,
                    'version': version,
                    'units': row['units'],
                    'speedup': c1_mean / row['comp_time'] if row['comp_time'] > 0 else 0
                })
                
    df_comp_speedup = pd.DataFrame(comp_speedup_rows)
    agg_comp_speedup = df_comp_speedup.groupby(['level', 'version', 'units', 'max_iter'], as_index=False)['speedup'].agg(
        speedup_mean='mean', speedup_min='min', speedup_max='max'
    )
    
    fig, axes = plt.subplots(1, len(levels), figsize=(6 * max(1, len(levels)), 5), sharey=True)
    if len(levels) == 1:
        axes = [axes]
        
    max_comp_sp = agg_comp_speedup['speedup_max'].max() if not agg_comp_speedup.empty else 1.0
    comp_sp_top_lim = max(4.5, float(np.ceil(max_comp_sp * 1.2)))  # e.g., 4.5 or 5.0

    for i, lvl in enumerate(levels):
        ax = axes[i]
        lvl_df = df[df['level'] == lvl]
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        lvl_data = agg_comp_speedup[(agg_comp_speedup['level'] == lvl) & (agg_comp_speedup['max_iter'] == latest_max_iter)]
        
        min_u = lvl_data['units'].min()
        max_u = lvl_data['units'].max()
        ax.plot([min_u, max_u], [min_u, max_u], linestyle='--', color='gray', label='Ideal Scaling', alpha=0.7)
        ax.axhline(1.0, color='black', linestyle=':', linewidth=1.2, alpha=0.55, label='Baseline (1 unit)')
        
        for version in available_versions:
            v_data = lvl_data[lvl_data['version'] == version].sort_values('units')
            if not v_data.empty:
                yerr_low = (v_data['speedup_mean'] - v_data['speedup_min']).to_numpy()
                yerr_high = (v_data['speedup_max'] - v_data['speedup_mean']).to_numpy()
                ax.errorbar(v_data['units'].to_numpy(), v_data['speedup_mean'].to_numpy(), 
                            yerr=[yerr_low, yerr_high],
                            marker=markers[version], color=colors[version], linewidth=2.2, markersize=8,
                            capsize=4, capthick=1.5,
                            label=labels[version])
                
        cells_str = f" (~{cell_counts[lvl]:,} cells)".replace(",", " ") if lvl in cell_counts else ""
        ax.set_title(f"Dynamic AMR: Level {lvl}{cells_str}\n({latest_max_iter} iterations)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Calculation Units (Threads / Processes)", fontsize=11, fontweight='semibold')
        if i == 0:
            ax.set_ylabel("Computation Speedup (relative to 1 unit)", fontsize=11, fontweight='semibold')
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(lvl_data['units'].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(bottom=0, top=comp_sp_top_lim)
        ax.legend(fontsize=9, loc='upper left')
        
    plt.suptitle("Advection 2D Dynamic AMR: Pure Computation Speedup (Excl. Adaptation)", fontsize=15, y=1.02, fontweight='bold')
    plt.tight_layout()
    out_comp_speedup_base = os.path.join(output_dir, "computation_speedup_comparison_mr_dynamic_naive")
    for ext in ["png", "pdf"]:
        plt.savefig(f"{out_comp_speedup_base}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved computation speedup plot to {out_comp_speedup_base}.png and {out_comp_speedup_base}.pdf")

    # =========================================================================
    # Plot 5: Adaptation Time Comparison vs. Units
    # =========================================================================
    fig, axes = plt.subplots(1, len(levels), figsize=(6 * max(1, len(levels)), 5), sharey=False)
    if len(levels) == 1:
        axes = [axes]

    for i, lvl in enumerate(levels):
        ax = axes[i]
        lvl_df = df[df['level'] == lvl]
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        lvl_data = agg_df[(agg_df['level'] == lvl) & (agg_df['max_iter'] == latest_max_iter)]

        for version in available_versions:
            v_data = lvl_data[lvl_data['version'] == version].sort_values('units')
            if not v_data.empty:
                yerr_low = (v_data['adapt_time_mean'] - v_data['adapt_time_min']).to_numpy()
                yerr_high = (v_data['adapt_time_max'] - v_data['adapt_time_mean']).to_numpy()
                ax.errorbar(v_data['units'].to_numpy(), v_data['adapt_time_mean'].to_numpy(), 
                            yerr=[yerr_low, yerr_high],
                            marker=markers[version], color=colors[version], linewidth=2.2, markersize=8,
                            capsize=4, capthick=1.5,
                            label=labels[version])
                
        cells_str = f" (~{cell_counts[lvl]:,} cells)".replace(",", " ") if lvl in cell_counts else ""
        ax.set_title(f"Dynamic AMR: Level {lvl}{cells_str}\n({latest_max_iter} iterations)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Calculation Units (Threads / Processes)", fontsize=11, fontweight='semibold')
        if i == 0:
            ax.set_ylabel("Adaptation Time (seconds)", fontsize=11, fontweight='semibold')
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(lvl_data['units'].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)
        
    plt.suptitle("Advection 2D Dynamic AMR: Mesh Adaptation Time Comparison", fontsize=15, y=1.02, fontweight='bold')
    plt.tight_layout()
    out_adapt_base = os.path.join(output_dir, "adaptation_time_comparison_mr_dynamic_naive")
    for ext in ["png", "pdf"]:
        plt.savefig(f"{out_adapt_base}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved adaptation time plot to {out_adapt_base}.png and {out_adapt_base}.pdf")

    # =========================================================================
    # Plot 6: Stacked Bar Charts (Computation vs. Adaptation Time Breakdown - Seconds)
    # =========================================================================
    fig, axes = plt.subplots(1, len(levels), figsize=(7.5 * max(1, len(levels)), 5.5), sharey=False)
    if len(levels) == 1:
        axes = [axes]

    for i, lvl in enumerate(levels):
        ax = axes[i]
        lvl_df = df[df['level'] == lvl]
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        lvl_data = agg_df[(agg_df['level'] == lvl) & (agg_df['max_iter'] == latest_max_iter)]
        
        unique_units = sorted(lvl_data['units'].unique())
        n_units = len(unique_units)
        n_versions = len(available_versions)
        
        x_indices = np.arange(n_units)
        bar_width = 0.8 / max(1, n_versions)
        
        for v_idx, version in enumerate(available_versions):
            v_data = lvl_data[lvl_data['version'] == version].set_index('units').reindex(unique_units)
            
            x_pos = x_indices - (0.8 / 2) + (v_idx + 0.5) * bar_width
            
            comp_vals = v_data['comp_time_mean'].fillna(0).to_numpy()
            adapt_vals = v_data['adapt_time_mean'].fillna(0).to_numpy()
            
            # Bottom bar: Computation (solid)
            ax.bar(x_pos, comp_vals, width=bar_width, color=colors[version], 
                   edgecolor='black', linewidth=0.8, alpha=0.9)
            
            # Top bar: Adaptation (hatched)
            ax.bar(x_pos, adapt_vals, width=bar_width, bottom=comp_vals, 
                   color=colors_adapt[version], hatch='//', edgecolor=colors[version], linewidth=0.8, alpha=0.85)

        cells_str = f" (~{cell_counts[lvl]:,} cells)".replace(",", " ") if lvl in cell_counts else ""
        ax.set_title(f"Dynamic AMR: Level {lvl}{cells_str}\n({latest_max_iter} iterations)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Calculation Units (Threads / Processes)", fontsize=11, fontweight='semibold')
        if i == 0:
            ax.set_ylabel("Execution Time Breakdown (seconds)", fontsize=11, fontweight='semibold')
        ax.set_xticks(x_indices)
        ax.set_xticklabels(unique_units)
        ax.set_ylim(bottom=0)

    # Build clean grouped legend
    legend_handles = []
    for version in available_versions:
        legend_handles.append(mpatches.Patch(facecolor=colors[version], edgecolor='black', label=f"{labels[version]} - Comp"))
        legend_handles.append(mpatches.Patch(facecolor=colors_adapt[version], hatch='//', edgecolor=colors[version], label=f"{labels[version]} - Adapt"))
        
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=len(available_versions), fontsize=10, frameon=True)

    fig.suptitle("Advection 2D Dynamic AMR: Computation vs. Adaptation Time Breakdown", fontsize=15, y=1.04, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_breakdown_base = os.path.join(output_dir, "time_breakdown_mr_dynamic_naive")
    for ext in ["png", "pdf"]:
        plt.savefig(f"{out_breakdown_base}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved time breakdown plot to {out_breakdown_base}.png and {out_breakdown_base}.pdf")

    # =========================================================================
    # Plot 7: Percentage Stacked Bar Charts (Adaptation vs Computation % of Runtime)
    # =========================================================================
    fig, axes = plt.subplots(1, len(levels), figsize=(7.5 * max(1, len(levels)), 5.5), sharey=True)
    if len(levels) == 1:
        axes = [axes]

    for i, lvl in enumerate(levels):
        ax = axes[i]
        lvl_df = df[df['level'] == lvl]
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        lvl_data = agg_df[(agg_df['level'] == lvl) & (agg_df['max_iter'] == latest_max_iter)]
        
        unique_units = sorted(lvl_data['units'].unique())
        n_units = len(unique_units)
        n_versions = len(available_versions)
        
        x_indices = np.arange(n_units)
        bar_width = 0.8 / max(1, n_versions)
        
        for v_idx, version in enumerate(available_versions):
            v_data = lvl_data[lvl_data['version'] == version].set_index('units').reindex(unique_units)
            
            x_pos = x_indices - (0.8 / 2) + (v_idx + 0.5) * bar_width
            
            comp_vals = v_data['comp_time_mean'].fillna(0).to_numpy()
            adapt_vals = v_data['adapt_time_mean'].fillna(0).to_numpy()
            total_vals = comp_vals + adapt_vals
            
            # Avoid division by zero
            pct_comp = np.where(total_vals > 0, (comp_vals / total_vals) * 100.0, 0.0)
            pct_adapt = np.where(total_vals > 0, (adapt_vals / total_vals) * 100.0, 0.0)
            
            # Bottom bar: Computation % (solid)
            ax.bar(x_pos, pct_comp, width=bar_width, color=colors[version], 
                   edgecolor='black', linewidth=0.8, alpha=0.9)
            
            # Top bar: Adaptation % (hatched)
            ax.bar(x_pos, pct_adapt, width=bar_width, bottom=pct_comp, 
                   color=colors_adapt[version], hatch='//', edgecolor=colors[version], linewidth=0.8, alpha=0.85)

        cells_str = f" (~{cell_counts[lvl]:,} cells)".replace(",", " ") if lvl in cell_counts else ""
        ax.set_title(f"Dynamic AMR: Level {lvl}{cells_str}\n({latest_max_iter} iterations)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Calculation Units (Threads / Processes)", fontsize=11, fontweight='semibold')
        if i == 0:
            ax.set_ylabel("Execution Time Ratio (%)", fontsize=11, fontweight='semibold')
        ax.set_xticks(x_indices)
        ax.set_xticklabels(unique_units)
        ax.set_ylim(bottom=0, top=105)
        ax.axhline(50, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=len(available_versions), fontsize=10, frameon=True)

    fig.suptitle("Advection 2D Dynamic AMR: Relative Time Share (% Comp vs % Adapt)", fontsize=15, y=1.04, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_pct_base = os.path.join(output_dir, "time_breakdown_percentage_mr_dynamic_naive")
    for ext in ["png", "pdf"]:
        plt.savefig(f"{out_pct_base}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved percentage breakdown plot to {out_pct_base}.png and {out_pct_base}.pdf")

if __name__ == "__main__":
    generate_comparison_plots()
