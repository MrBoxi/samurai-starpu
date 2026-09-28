#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common import ROOT_PROJECT_PATH, PERF_DIR

def plot_variant_granularity(df, label, levels, cell_counts_dict, title_prefix, filename_prefix):
    variant_df = df[(df['label'] == label) & (df['nb_tasks'] < 512)]
    if variant_df.empty:
        print(f"Warning: No data found for {label}")
        return
        
    agg_df = variant_df.groupby(['level', 'nb_tasks', 'max_iter'], as_index=False)['time'].agg(
        time_mean='mean', time_min='min', time_max='max'
    )
    
    color = "#1f77b4" if "uniform" in label else "#2ca02c"
    marker = "o" if "uniform" in label else "^"
    
    # ------------------ Plot: Execution Time vs. Number of Tasks ------------------
    fig, axes = plt.subplots(1, len(levels), figsize=(18, 5.5), sharey=False)
    if len(levels) == 1:
        axes = [axes]
        
    for i, lvl in enumerate(levels):
        ax = axes[i]
        lvl_df = variant_df[variant_df['level'] == lvl]
        if lvl_df.empty:
            continue
        latest_max_iter = lvl_df['max_iter'].iloc[-1]
        v_data = agg_df[(agg_df['level'] == lvl) & (agg_df['max_iter'] == latest_max_iter)].sort_values('nb_tasks')
        
        # Identify baseline (the very first point: 16 tasks = 1 task per thread)
        first_row = v_data[v_data['nb_tasks'] == 16]
        if first_row.empty:
            first_row = v_data.iloc[0:1]
        t_first = first_row['time_mean'].values[0]
        
        # Horizontal reference line on the very first point
        ax.axhline(t_first, linestyle='--', color='#d62728', linewidth=1.8, alpha=0.85, 
                   label=f'Baseline 16 tasks ({t_first:.3f} s)')
        
        # Plot execution time with [min, max] error bars
        yerr_low = v_data['time_mean'] - v_data['time_min']
        yerr_high = v_data['time_max'] - v_data['time_mean']
        ax.errorbar(v_data['nb_tasks'], v_data['time_mean'], 
                    yerr=[yerr_low, yerr_high],
                    marker=marker, color=color, linewidth=2, markersize=8,
                    capsize=4, capthick=1.5,
                    label=title_prefix)
        
        c_str = cell_counts_dict.get(lvl, "")
        ax.set_title(f"Level {lvl} ({c_str} cells)\n({latest_max_iter} iterations)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Number of Tasks / Subdomains", fontsize=12)
        if i == 0:
            ax.set_ylabel("Execution Time (seconds)", fontsize=12)
            
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(v_data['nb_tasks'].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        y_max_val = max(v_data['time_max'].max(), t_first)
        ax.set_ylim(bottom=0, top=y_max_val * 1.30)
        ax.legend(fontsize=11)
        
    plt.suptitle(f"Impact of Granularity on Execution Time (Fixed 16 Threads): {title_prefix}", fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(ROOT_PROJECT_PATH, f"{filename_prefix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

def generate_granularity_plots():
    csv_file = os.path.join(PERF_DIR, "perf_results_granularity.csv")
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Cannot plot.")
        return
        
    df = pd.read_csv(csv_file)
    
    # Ignore 512 tasks data
    df = df[df['nb_tasks'] < 512]
    
    sns.set_theme(style="whitegrid")
    
    # 1. Plot Uniform Granularity (Levels 11, 12, 13)
    uniform_cells = {11: "4 194 304", 12: "16 777 216", 13: "67 108 864"}
    plot_variant_granularity(df, "starpu_uniform", [11, 12, 13], uniform_cells, 
                             "StarPU Uniform", "granularity_uniform")
                             
    # 2. Plot MR Static Granularity (Levels 13, 14, 15)
    mr_cells = {13: "403 096", 14: "805 256", 15: "1 608 072"}
    plot_variant_granularity(df, "starpu_mr_static", [13, 14, 15], mr_cells, 
                             "StarPU MR Static", "granularity_mr_static")

if __name__ == "__main__":
    generate_granularity_plots()
