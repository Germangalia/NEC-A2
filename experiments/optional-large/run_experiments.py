#!/usr/bin/env python3
"""
Optional Algorithm (Simulated Annealing) experiments for JSSP - Large Dataset.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.insert(0, project_root)

import random
import csv
import datetime

from shared.parser import get_instance_by_name
from shared.encoding import create_random_chromosome
from shared.fitness import compute_makespan
import importlib.util, math

# Load operators
spec = importlib.util.spec_from_file_location('operators', os.path.join(project_root, 'genetic-algorithm', 'operators.py'))
operators = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operators)

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

def simulated_annealing(instance, config):
    """Run SA with given configuration."""
    max_iter = config['max_iterations']
    temp = config['initial_temperature']
    cooling = config['cooling_rate']
    mutation = config['mutation']

    print(f"\n{'='*60}")
    print(f"Starting SA with configuration:")
    print(f"  Max iterations: {max_iter}")
    print(f"  Initial temperature: {temp}")
    print(f"  Cooling rate: {cooling}")
    print(f"  Mutation: {mutation}")
    print(f"{'='*60}")

    current = create_random_chromosome(instance)
    current_cost = compute_makespan(instance, current)
    best = current.copy()
    best_cost = current_cost
    best_history = [best_cost]
    
    for i in range(max_iter):
        if mutation == 'swap':
            neighbor = operators.swap_mutation(current)
        else:
            neighbor = operators.inversion_mutation(current)
        neighbor_cost = compute_makespan(instance, neighbor)
        
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost
        
        best_history.append(best_cost)
        temp *= cooling
        
        if i % 1000 == 0:
            print(f"Iteration {i:5d}: Best = {best_cost:4d}, Temp = {temp:.2f}")
    
    print(f"\n{'='*60}")
    print(f"SA completed!")
    print(f"Final best makespan: {best_cost}")
    print(f"{'='*60}")

    return best, best_cost, best_history

def main():
    """Run SA experiments and save results."""
    results_dir = os.path.join(project_root, 'results/optional-algorithm/large')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(results_dir, f'experiment_log_{timestamp}.txt')

    original_stdout = sys.stdout
    logger = Logger(log_file)
    sys.stdout = logger

    print("="*80)
    print(f"Optional Algorithm Experiments Log - Large Dataset (abz7) - {timestamp}")
    print("="*80)

    configs = [
        {'max_iterations': 5000, 'initial_temperature': 1000, 'cooling_rate': 0.995, 'mutation': 'swap'},
        {'max_iterations': 5000, 'initial_temperature': 1000, 'cooling_rate': 0.995, 'mutation': 'inversion'},
        {'max_iterations': 10000, 'initial_temperature': 1500, 'cooling_rate': 0.99, 'mutation': 'swap'},
    ]

    all_results = []

    print("\nLoading instance...")
    instance = get_instance_by_name(os.path.join(project_root, 'dataset/jobshop1.txt'), 'abz7')
    print(f"Loaded: {instance.num_jobs} jobs, {instance.num_machines} machines")

    for exp_id, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"Experiment {exp_id}/{len(configs)}")
        print(f"Config: {config['mutation']}, iter={config['max_iterations']}, temp={config['initial_temperature']}")
        print(f"{'='*80}")

        try:
            best_chrom, best_ms, history = simulated_annealing(instance, config)

            result = {
                'experiment_id': exp_id,
                'instance': 'abz7',
                'max_iterations': config['max_iterations'],
                'initial_temperature': config['initial_temperature'],
                'cooling_rate': config['cooling_rate'],
                'mutation': config['mutation'],
                'best_makespan': best_ms,
                'success': True,
                'error': '',
                'history': history
            }

            all_results.append(result)

        except Exception as e:
            import traceback
            print(f"\n✗ Error occurred:")
            print(f"Error: {e}")
            print(f"Traceback:")
            print(traceback.format_exc())
            result = {
                'experiment_id': exp_id,
                'instance': 'abz7',
                'max_iterations': config['max_iterations'],
                'initial_temperature': config['initial_temperature'],
                'cooling_rate': config['cooling_rate'],
                'mutation': config['mutation'],
                'best_makespan': None,
                'success': False,
                'error': str(e)
            }
            all_results.append(result)

    sys.stdout = original_stdout

    summary_file = os.path.join(results_dir, f'experiment_summary_{timestamp}.csv')
    with open(summary_file, 'w', newline='') as f:
        fieldnames = ['experiment_id', 'instance', 'max_iterations',
                     'initial_temperature', 'cooling_rate', 'mutation', 'best_makespan', 'success', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Write results without 'history' field
        csv_results = [{k: v for k, v in r.items() if k != 'history'} for r in all_results]
        writer.writerows(csv_results)

    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Log: {log_file}")
    print(f"Summary: {summary_file}")
    print(f"{'='*80}")

    print("\nResults Summary:")
    print(f"{'ID':<3} {'Iter':<6} {'Temp':<6} {'Mut':<10} {'Best':<6} {'Status':<8}")
    print("-" * 50)
    for r in all_results:
        makespan = r['best_makespan'] if r['best_makespan'] else "N/A"
        status = "✓" if r['success'] else "✗"
        print(f"{r['experiment_id']:<3} {r['max_iterations']:<6} {r['initial_temperature']:<6} {r['mutation']:<10} {makespan:<6} {status:<8}")

    # Generate plots
    print(f"\n{'='*80}")
    print(f"Generating evolution plots...")
    print(f"{'='*80}")
    
    import matplotlib.pyplot as plt
    
    # Individual plots
    fig, axes = plt.subplots(1, len(all_results), figsize=(6*len(all_results), 4))
    if len(all_results) == 1:
        axes = [axes]
    
    global_best = min([r['best_makespan'] for r in all_results if r['best_makespan']])
    
    for i, result in enumerate(all_results):
        ax = axes[i]
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
    plot_file = os.path.join(results_dir, f'sa_evolution_plots_abz7_{timestamp}.png')
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
    ax.set_title('Simulated Annealing: Comparison of All Experiments (abz7)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    comparison_file = os.path.join(results_dir, f'sa_evolution_comparison_abz7_{timestamp}.png')
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nEvolution plots generated:")
    print(f"  - {plot_file}")
    print(f"  - {comparison_file}")

if __name__ == '__main__':
    main()
