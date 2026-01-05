#!/usr/bin/env python3
"""
Simplified GA implementation for testing and generating results.
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "genetic-algorithm"))

import random
import csv
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from shared.parser import get_instance_by_name
from shared.encoding import create_random_chromosome
from shared.fitness import compute_makespan
import importlib.util
spec = importlib.util.spec_from_file_location('operators', os.path.join(project_root, 'genetic-algorithm', 'operators.py'))
operators = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operators)
crossover_ox = operators.crossover_ox
crossover_pmx = operators.crossover_pmx
mutate_swap = operators.mutate_swap
mutate_inversion = operators.mutate_inversion
tournament_selection = operators.tournament_selection
rank_selection = operators.rank_selection

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

def simple_ga(instance, config):
    """Run a simple GA with given configuration."""
    population_size = config['population_size']
    max_generations = config['max_generations']
    crossover_rate = config['crossover_rate']
    mutation_rate = config['mutation_rate']
    elite_size = config['elite_size']
    selection_method = config['selection']
    crossover_method = config['crossover']
    mutation_method = config['mutation']

    print(f"\n{'='*60}")
    print(f"Starting GA with configuration:")
    print(f"  Population size: {population_size}")
    print(f"  Max generations: {max_generations}")
    print(f"  Selection: {selection_method}")
    print(f"  Crossover: {crossover_method} (rate: {crossover_rate})")
    print(f"  Mutation: {mutation_method} (rate: {mutation_rate})")
    print(f"  Elite size: {elite_size}")
    print(f"{'='*60}")

    # Initialize population
    print("\nInitializing population...")
    population = [create_random_chromosome(instance) for _ in range(population_size)]
    print(f"Population initialized: {len(population)} chromosomes")

    best_makespan_history = []
    best_chromosome = None
    best_makespan = float('inf')

    for generation in range(max_generations):
        # Evaluate
        fitness = [compute_makespan(instance, chrom) for chrom in population]

        # Track best
        min_fitness = min(fitness)
        avg_fitness = sum(fitness) / len(fitness)
        best_makespan_history.append(min_fitness)

        if min_fitness < best_makespan:
            best_makespan = min_fitness
            best_idx = fitness.index(min_fitness)
            best_chromosome = population[best_idx].copy()
            print(f"  New best found at generation {generation}: {best_makespan}")

        # Print progress every generation
        print(f"Generation {generation:3d}: Best = {min_fitness:3d}, Avg = {avg_fitness:6.2f}")

        # Selection
        num_parents = population_size - elite_size
        parents = []
        for _ in range(num_parents):
            if selection_method == 'tournament':
                parent = tournament_selection(population, fitness, 3)
            else:
                parent = rank_selection(population, fitness)
            parents.append(parent)

        # Crossover
        if crossover_method == 'ox':
            offspring = crossover_ox(parents, crossover_rate)
        else:
            offspring = crossover_pmx(parents, crossover_rate)

        # Mutation
        if mutation_method == 'swap':
            offspring = mutate_swap(offspring, mutation_rate)
        else:
            offspring = mutate_inversion(offspring, mutation_rate)

        # Elitism
        elite_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])[:elite_size]
        elite = [population[i].copy() for i in elite_indices]

        # New population
        population = offspring + elite

    print(f"\n{'='*60}")
    print(f"GA completed!")
    print(f"Final best makespan: {best_makespan}")
    print(f"{'='*60}")

    return best_chromosome, best_makespan, best_makespan_history

def generate_evolution_plots(all_results, all_histories, results_dir, instance_name, timestamp):
    """Generate evolution plots for all experiments."""
    
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

def main():
    """Run experiments and save results."""
    results_dir = '../../results/large'
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(results_dir, f'experiment_log_{timestamp}.txt')

    # Redirect stdout to capture all output
    original_stdout = sys.stdout
    logger = Logger(log_file)
    sys.stdout = logger

    print("="*80)
    print(f"GA Experiments Log - {timestamp}")
    print("="*80)

    # Load instance
    print("\nLoading instance...")
    instance = get_instance_by_name(os.path.join(project_root, 'dataset/jobshop1.txt'), 'abz7')
    print(f"Loaded: {instance.num_jobs} jobs, {instance.num_machines} machines")

    # Experiment configurations
    experiments = [
        {'population_size': 30, 'max_generations': 50, 'selection': 'tournament', 'crossover': 'ox', 'mutation': 'swap', 'crossover_rate': 0.8, 'mutation_rate': 0.1, 'elite_size': 2},
        {'population_size': 50, 'max_generations': 100, 'selection': 'tournament', 'crossover': 'pmx', 'mutation': 'swap', 'crossover_rate': 0.8, 'mutation_rate': 0.1, 'elite_size': 2},
        {'population_size': 30, 'max_generations': 50, 'selection': 'rank', 'crossover': 'ox', 'mutation': 'inversion', 'crossover_rate': 0.8, 'mutation_rate': 0.1, 'elite_size': 2},
        {'population_size': 50, 'max_generations': 100, 'selection': 'rank', 'crossover': 'pmx', 'mutation': 'inversion', 'crossover_rate': 0.8, 'mutation_rate': 0.1, 'elite_size': 2},
        {'population_size': 40, 'max_generations': 75, 'selection': 'tournament', 'crossover': 'ox', 'mutation': 'inversion', 'crossover_rate': 0.8, 'mutation_rate': 0.1, 'elite_size': 2},
        {'population_size': 40, 'max_generations': 75, 'selection': 'rank', 'crossover': 'pmx', 'mutation': 'swap', 'crossover_rate': 0.8, 'mutation_rate': 0.1, 'elite_size': 2},
    ]

    all_results = []
    all_histories = []

    print(f"\n{'='*80}")
    print(f"Running {len(experiments)} experiments")
    print(f"{'='*80}")

    for i, config in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"Experiment {i}/{len(experiments)}")
        print(f"Config: {config['selection']}/{config['crossover']}/{config['mutation']}")
        print(f"Population: {config['population_size']}, Generations: {config['max_generations']}")
        print(f"{'='*80}")

        try:
            best_chrom, best_ms, history = simple_ga(instance, config)
            all_histories.append(history)

            result = {
                'experiment_id': i,
                'instance': 'abz7',
                'population_size': config['population_size'],
                'max_generations': config['max_generations'],
                'selection': config['selection'],
                'crossover': config['crossover'],
                'mutation': config['mutation'],
                'best_makespan': best_ms,
                'success': True,
                'error': ''
            }

            print(f"\nEvolution history: {history}")

        except Exception as e:
            import traceback
            print(f"\n✗ Error occurred:")
            print(f"Error: {e}")
            print(f"Traceback:")
            print(traceback.format_exc())
            result = {
                'experiment_id': i,
                'instance': 'abz7',
                'population_size': config['population_size'],
                'max_generations': config['max_generations'],
                'selection': config['selection'],
                'crossover': config['crossover'],
                'mutation': config['mutation'],
                'best_makespan': None,
                'success': False,
                'error': str(e)
            }
            all_histories.append([])

        all_results.append(result)

    # Restore stdout
    sys.stdout = original_stdout
    
    # Generate evolution plots
    print(f"\n{'='*80}")
    print("Generating evolution plots...")
    print(f"{'='*80}")
    generate_evolution_plots(all_results, all_histories, results_dir, 'abz7', timestamp)

    # Save summary CSV
    summary_file = os.path.join(results_dir, f'experiment_summary_{timestamp}.csv')
    with open(summary_file, 'w', newline='') as f:
        fieldnames = ['experiment_id', 'instance', 'population_size', 'max_generations',
                     'selection', 'crossover', 'mutation', 'best_makespan', 'success', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Log: {log_file}")
    print(f"Summary: {summary_file}")
    print(f"{'='*80}")

    # Print summary table
    print("\nResults Summary:")
    print(f"{'ID':<3} {'Pop':<5} {'Gen':<5} {'Sel':<10} {'Xover':<6} {'Mut':<10} {'Best':<8} {'Status':<8}")
    print("-" * 80)
    for r in all_results:
        makespan = r['best_makespan'] if r['best_makespan'] else "N/A"
        status = "✓" if r['success'] else "✗"
        print(f"{r['experiment_id']:<3} {r['population_size']:<5} "
              f"{r['max_generations']:<5} {r['selection']:<10} {r['crossover']:<6} "
              f"{r['mutation']:<10} {makespan:<8} {status:<8}")

if __name__ == '__main__':
    main()