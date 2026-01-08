#!/usr/bin/env python3
"""
Shared utilities for running experiments.
"""

import sys
import os
import csv
import datetime
import matplotlib.pyplot as plt


class Logger:
    """Logger that captures all output to both console and file."""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        
    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, 'a') as f:
            f.write(message)
    
    def flush(self):
        self.terminal.flush()


def setup_experiment(results_subdir, experiment_type):
    """
    Setup experiment environment.
    
    Args:
        results_subdir: Subdirectory for results (e.g., 'small', 'medium', 'large')
        experiment_type: Type of experiment ('ga' or 'sa')
    
    Returns:
        tuple: (results_dir, timestamp, log_file, logger, original_stdout)
    """
    # Get the absolute path of the shared directory and navigate to project root
    shared_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(shared_dir)
    
    results_dir = os.path.join(project_root, f'results/{results_subdir}')
    
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(results_dir, f'experiment_log_{timestamp}.txt')
    
    original_stdout = sys.stdout
    logger = Logger(log_file)
    sys.stdout = logger
    
    return results_dir, timestamp, log_file, logger, original_stdout


def save_results_csv(results, results_dir, timestamp, exclude_fields=None):
    """
    Save experiment results to CSV file.
    
    Args:
        results: List of result dictionaries
        results_dir: Directory to save results
        timestamp: Timestamp for filename
        exclude_fields: List of field names to exclude from CSV
    """
    if exclude_fields is None:
        exclude_fields = []
    
    # Get fieldnames from first result, excluding specified fields
    if results:
        fieldnames = [k for k in results[0].keys() if k not in exclude_fields]
    else:
        fieldnames = []
    
    summary_file = os.path.join(results_dir, f'experiment_summary_{timestamp}.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write results excluding specified fields
        csv_results = [{k: v for k, v in r.items() if k not in exclude_fields} for r in results]
        writer.writerows(csv_results)
    
    return summary_file


def print_ga_summary(results):
    """Print GA experiment results summary."""
    print("\nResults Summary:")
    print(f"{'ID':<3} {'Pop':<5} {'Gen':<5} {'Sel':<10} {'Xover':<6} {'Mut':<10} {'Best':<8} {'Status':<8}")
    print("-" * 80)
    for r in results:
        makespan = r['best_makespan'] if r['best_makespan'] else "N/A"
        status = "✓" if r['success'] else "✗"
        print(f"{r['experiment_id']:<3} {r['population_size']:<5} "
              f"{r['max_generations']:<5} {r['selection']:<10} {r['crossover']:<6} "
              f"{r['mutation']:<10} {makespan:<8} {status:<8}")


def print_sa_summary(results):
    """Print SA experiment results summary."""
    print("\nResults Summary:")
    print(f"{'ID':<3} {'Iter':<6} {'Temp':<6} {'Mut':<10} {'Best':<6} {'Status':<8}")
    print("-" * 50)
    for r in results:
        makespan = r['best_makespan'] if r['best_makespan'] else "N/A"
        status = "✓" if r['success'] else "✗"
        print(f"{r['experiment_id']:<3} {r['max_iterations']:<6} {r['initial_temperature']:<6} "
              f"{r['mutation']:<10} {makespan:<6} {status:<8}")


def generate_ga_evolution_plots(all_results, all_histories, results_dir, instance_name, timestamp):
    """Generate evolution plots for GA experiments."""
    
    # Create figure with subplots for all experiments
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'GA Evolution Plots - {instance_name.upper()}', fontsize=16, fontweight='bold')
    
    # Find best solution across all experiments
    all_best = [min(history) if history else float('inf') for history in all_histories]
    global_best = min(all_best) if all_best else None
    
    # Plot each experiment
    for idx, (result, history) in enumerate(zip(all_results, all_histories)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        if not history or not result['success']:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue
        
        generations = list(range(len(history)))
        
        # Plot evolution
        ax.plot(generations, history, 'b-', linewidth=2, label='Best Makespan')
        ax.fill_between(generations, history, alpha=0.3)
        
        # Mark the best solution
        best_value = min(history)
        best_gen = history.index(best_value)
        ax.scatter([best_gen], [best_value], color='red', s=100, zorder=5, 
                  label=f'Best: {best_value}')
        
        # Highlight if this is the global best
        if best_value == global_best:
            ax.set_title(f"Exp {result['experiment_id']}: {result['selection']}/{result['crossover']}/{result['mutation']}\n★ GLOBAL BEST ({best_value})", 
                        fontweight='bold', color='darkgreen')
        else:
            ax.set_title(f"Exp {result['experiment_id']}: {result['selection']}/{result['crossover']}/{result['mutation']}\nBest: {best_value}")
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Makespan')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(results_dir, f'evolution_plots_{instance_name}_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a comparison plot showing all experiments together
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(range(len(all_histories)))
    
    for idx, (result, history) in enumerate(zip(all_results, all_histories)):
        if not history or not result['success']:
            continue
        
        generations = list(range(len(history)))
        label = f"Exp {result['experiment_id']}: {result['selection']}/{result['crossover']}/{result['mutation']} (Best: {min(history)})"
        ax.plot(generations, history, linewidth=2, color=colors[idx], label=label, alpha=0.8)
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Makespan', fontsize=12)
    ax.set_title(f'GA Evolution Comparison - {instance_name.upper()}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save comparison plot
    output_file = os.path.join(results_dir, f'evolution_comparison_{instance_name}_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nEvolution plots generated:")
    print(f"  - {output_file}")
    print(f"  - {os.path.join(results_dir, f'evolution_plots_{instance_name}_{timestamp}.png')}")


def generate_sa_evolution_plots(all_results, results_dir, instance_name, timestamp):


    """Generate evolution plots for SA experiments."""


    


    # Individual plots - 2 rows, 3 columns


    fig, axes = plt.subplots(2, 3, figsize=(18, 10))


    fig.suptitle(f'SA Evolution Plots - {instance_name.upper()}', fontsize=16, fontweight='bold')


    


    global_best = min([r['best_makespan'] for r in all_results if r['best_makespan']])


    


    for i, result in enumerate(all_results):


        row = i // 3


        col = i % 3


        ax = axes[row, col]


        history = result.get('history', [])


        iterations = range(len(history))


        


        ax.plot(iterations, history, linewidth=2, color='#2E86AB')


        ax.set_xlabel('Iteration', fontsize=12)


        ax.set_ylabel('Best Makespan', fontsize=12)


        ax.set_title(f"Exp {result['experiment_id']}: {result['mutation']}, iter={result['max_iterations']}\nBest: {result['best_makespan']}", fontsize=11)


        ax.grid(True, alpha=0.3)


        


        if result['best_makespan'] == global_best:


            ax.set_facecolor('#E8F4F8')


            ax.set_title(f"Exp {result['experiment_id']}: {result['mutation']}, iter={result['max_iterations']}\nBest: {result['best_makespan']} ★", fontsize=11)


    


    plt.tight_layout()


    plot_file = os.path.join(results_dir, f'sa_evolution_plots_{instance_name}_{timestamp}.png')


    plt.savefig(plot_file, dpi=150, bbox_inches='tight')


    plt.close()
    
    # Comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for result in all_results:
        history = result.get('history', [])
        iterations = range(len(history))
        label = f"Exp {result['experiment_id']}: {result['mutation']}, iter={result['max_iterations']}"
        color = '#2E86AB' if result['best_makespan'] == global_best else '#A23B72'
        linewidth = 2.5 if result['best_makespan'] == global_best else 1.5
        ax.plot(iterations, history, label=label, linewidth=linewidth, color=color)
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Best Makespan', fontsize=14)
    ax.set_title(f'Simulated Annealing: Comparison of All Experiments ({instance_name.upper()})', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    comparison_file = os.path.join(results_dir, f'sa_evolution_comparison_{instance_name}_{timestamp}.png')
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nEvolution plots generated:")
    print(f"  - {plot_file}")
    print(f"  - {comparison_file}")
