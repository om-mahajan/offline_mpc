#!/usr/bin/env python3
"""
Aggregate evaluation results across all seeds and tasks
Creates summary tables and plots
"""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def collect_results(base_dir, tasks, seeds):
    """
    Collect all evaluation results into a structured format
    """
    results = []
    
    for task in tasks:
        task_path = Path(base_dir).parent / task.replace('OfflinePointGoal2Gymnasium-v0', task)
        
        for seed in seeds:
            seed_pattern = f"seed-{seed:03d}-*"
            seed_dirs = list(task_path.glob(seed_pattern))
            
            if not seed_dirs:
                print(f"Warning: No directory found for {task}, seed {seed}")
                continue
            
            seed_dir = seed_dirs[0]
            result_file = seed_dir / 'evaluation_results' / f'eval_results_seed{seed}.json'
            
            if not result_file.exists():
                print(f"Warning: No results file found at {result_file}")
                continue
            
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            stats = data['statistics']
            results.append({
                'task': task,
                'seed': seed,
                'mean_reward': stats['mean_reward'],
                'std_reward': stats['std_reward'],
                'mean_cost': stats['mean_cost'],
                'std_cost': stats['std_cost'],
                'mean_length': stats['mean_length'],
                'std_length': stats['std_length'],
                'num_episodes': stats['num_episodes']
            })
    
    return pd.DataFrame(results)

def create_summary_table(df):
    """
    Create summary statistics grouped by task
    """
    summary = df.groupby('task').agg({
        'mean_reward': ['mean', 'std'],
        'mean_cost': ['mean', 'std'],
        'mean_length': ['mean', 'std']
    })
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    return summary

def plot_results(df, output_dir):
    """
    Create visualization plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Plot 1: Reward comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Reward
    ax = axes[0]
    for task in df['task'].unique():
        task_data = df[df['task'] == task]
        ax.plot(task_data['seed'], task_data['mean_reward'], 'o-', label=task, linewidth=2, markersize=8)
        ax.fill_between(
            task_data['seed'],
            task_data['mean_reward'] - task_data['std_reward'],
            task_data['mean_reward'] + task_data['std_reward'],
            alpha=0.2
        )
    ax.set_xlabel('Seed', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Reward Across Seeds', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cost
    ax = axes[1]
    for task in df['task'].unique():
        task_data = df[df['task'] == task]
        ax.plot(task_data['seed'], task_data['mean_cost'], 'o-', label=task, linewidth=2, markersize=8)
        ax.fill_between(
            task_data['seed'],
            task_data['mean_cost'] - task_data['std_cost'],
            task_data['mean_cost'] + task_data['std_cost'],
            alpha=0.2
        )
    ax.set_xlabel('Seed', fontsize=12)
    ax.set_ylabel('Mean Cost', fontsize=12)
    ax.set_title('Cost Across Seeds', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Length
    ax = axes[2]
    for task in df['task'].unique():
        task_data = df[df['task'] == task]
        ax.plot(task_data['seed'], task_data['mean_length'], 'o-', label=task, linewidth=2, markersize=8)
    ax.set_xlabel('Seed', fontsize=12)
    ax.set_ylabel('Mean Episode Length', fontsize=12)
    ax.set_title('Episode Length Across Seeds', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_dir / 'evaluation_summary.png'}")
    
    # Plot 2: Box plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    df.boxplot(column='mean_reward', by='task', ax=ax)
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Reward Distribution by Task', fontsize=14, fontweight='bold')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')
    
    ax = axes[1]
    df.boxplot(column='mean_cost', by='task', ax=ax)
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Mean Cost', fontsize=12)
    ax.set_title('Cost Distribution by Task', fontsize=14, fontweight='bold')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_dir / 'evaluation_boxplots.png'}")

def main(args):
    print("="*60)
    print("Aggregating Evaluation Results")
    print("="*60)
    
    # Collect results
    print("\nCollecting results...")
    df = collect_results(args.base_dir, args.tasks, args.seeds)
    
    if df.empty:
        print("Error: No results found!")
        return
    
    print(f"✓ Collected results for {len(df)} experiments")
    
    # Create summary
    print("\nCreating summary statistics...")
    summary = create_summary_table(df)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(summary.to_string())
    print("="*60)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / 'all_results.csv', index=False)
    summary.to_csv(output_dir / 'summary_statistics.csv')
    
    print(f"\n✓ Saved detailed results: {output_dir / 'all_results.csv'}")
    print(f"✓ Saved summary: {output_dir / 'summary_statistics.csv'}")
    
    # Create plots
    if not args.no_plots:
        print("\nGenerating plots...")
        plot_results(df, output_dir)
    
    print("\n" + "="*60)
    print("Aggregation complete!")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate evaluation results')
    
    parser.add_argument('--base-dir', type=str, required=True,
                       help='Base directory containing model checkpoints')
    parser.add_argument('--tasks', nargs='+', 
                       default=['OfflinePointGoal2Gymnasium-v0', 'OfflinePointButton1Gymnasium-v0'],
                       help='List of tasks to aggregate')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                       help='List of seeds to aggregate')
    parser.add_argument('--output-dir', type=str, default='./aggregated_results',
                       help='Directory to save aggregated results')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    main(args)
