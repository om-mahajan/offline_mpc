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

def collect_results(base_dir, tasks, seeds, algo_name="ipl_flow_matching_v2"):
    """
    Collect results from new inference.py output.
    Looks for:
        base_dir/TASK/IPLV4/merged/TASK/ALGO/seed-XXX-*/evaluation_results_all/all_checkpoint_results_seedX.json
    """
    results = []

    for task in tasks:
        task_path = Path(base_dir) / task / "IPLV4" / "merged" / task / algo_name

        if not task_path.exists():
            print(f"Warning: Task path not found: {task_path}")
            continue

        # For each seed
        for seed in seeds:
            seed_pattern = f"seed-{seed:03d}-*"
            seed_dirs = list(task_path.glob(seed_pattern))

            if not seed_dirs:
                print(f"Warning: No seed directory found for {task}, seed {seed}")
                continue

            seed_dir = seed_dirs[0]

            result_file = seed_dir / "evaluation_results_all" / f"all_checkpoint_results_seed{seed}.json"

            if not result_file.exists():
                print(f"Warning: Missing: {result_file}")
                continue

            with open(result_file, "r") as f:
                checkpoint_list = json.load(f)

            # checkpoint_list is a list of dicts
            for entry in checkpoint_list:
                results.append({
                    "task": task,
                    "seed": seed,
                    "iteration": entry["iteration"],
                    "mean_reward": entry["mean_reward"],
                    "std_reward": entry["std_reward"],
                    "mean_cost": entry["mean_cost"],
                    "std_cost": entry["std_cost"],
                    "mean_length": entry["mean_length"],
                    "std_length": entry["std_length"],
                })

    return pd.DataFrame(results)

def create_summary_table(df):
    """
    Summary statistics grouped by task and training iteration.
    """
    summary = df.groupby(["task", "iteration"]).agg({
        "mean_reward": ["mean", "std"],
        "mean_cost": ["mean", "std"],
        "mean_length": ["mean", "std"],
    })

    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    return summary

def plot_results(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # ======================================================
    # 1. Reward Curve vs Iteration
    # ======================================================
    plt.figure(figsize=(10, 6))
    for task in df["task"].unique():
        task_df = df[df["task"] == task].sort_values("iteration")
        sns.lineplot(
            x="iteration", y="mean_reward", hue="seed",
            style="seed", markers=True, data=task_df, legend=False
        )
    plt.title("Reward vs Training Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "reward_vs_iteration.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'reward_vs_iteration.png'}")

    # ======================================================
    # 2. Cost Curve vs Iteration
    # ======================================================
    plt.figure(figsize=(10, 6))
    for task in df["task"].unique():
        task_df = df[df["task"] == task].sort_values("iteration")
        sns.lineplot(
            x="iteration", y="mean_cost", hue="seed",
            style="seed", markers=True, data=task_df, legend=False
        )
    plt.title("Cost vs Training Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "cost_vs_iteration.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'cost_vs_iteration.png'}")

    # ======================================================
    # 3. Episode Length Curve vs Iteration
    # ======================================================
    plt.figure(figsize=(10, 6))
    for task in df["task"].unique():
        task_df = df[df["task"] == task].sort_values("iteration")
        sns.lineplot(
            x="iteration", y="mean_length", hue="seed",
            style="seed", markers=True, data=task_df, legend=False
        )
    plt.title("Episode Length vs Training Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Episode Length")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "length_vs_iteration.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'length_vs_iteration.png'}")

def main(args):
    print("="*60)
    print("Aggregating Evaluation Results")
    print("="*60)
    
    # Collect results
    print("\nCollecting results...")
    df = collect_results(args.base_dir, args.tasks, args.seeds, args.algo_name)
    
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
                       help='Base log directory (e.g., ~/safe_diff/offline_mpc/logs/merged)')
    parser.add_argument('--tasks', nargs='+', 
                       default=['OfflinePointGoal2Gymnasium-v0', 'OfflinePointButton1Gymnasium-v0'],
                       help='List of tasks to aggregate')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                       help='List of seeds to aggregate')
    parser.add_argument('--algo-name', type=str, default='ipl_flow_matching_v2',
                       help='Algorithm directory name')
    parser.add_argument('--output-dir', type=str, default='./aggregated_results',
                       help='Directory to save aggregated results')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    main(args)
