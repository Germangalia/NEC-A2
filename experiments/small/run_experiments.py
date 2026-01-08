#!/usr/bin/env python3
"""
GA experiments for JSSP - Small Dataset (ft06).
"""

import sys
import os
# Get the absolute path of the experiments directory and navigate to project root
experiments_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(experiments_dir))
sys.path.insert(0, project_root)

import traceback

from shared.parser import get_instance_by_name
from shared.ga_algorithm import simple_ga
from shared.experiment_utils import (
    setup_experiment, save_results_csv, print_ga_summary, generate_ga_evolution_plots
)


def main():
    """Run GA experiments and save results."""
    # Setup experiment
    results_dir, timestamp, log_file, logger, original_stdout = setup_experiment('small', 'ga')
    
    print("="*80)
    print(f"GA Experiments Log - {timestamp}")
    print("="*80)

    # Load instance
    print("\nLoading instance...")
    instance = get_instance_by_name(os.path.join(project_root, 'dataset/jobshop1.txt'), 'ft06')
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
                'instance': 'ft06',
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
            print(f"\n✗ Error occurred:")
            print(f"Error: {e}")
            print(f"Traceback:")
            print(traceback.format_exc())
            result = {
                'experiment_id': i,
                'instance': 'ft06',
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
    generate_ga_evolution_plots(all_results, all_histories, results_dir, 'ft06', timestamp)

    # Save summary CSV
    summary_file = save_results_csv(all_results, results_dir, timestamp)

    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Log: {log_file}")
    print(f"Summary: {summary_file}")
    print(f"{'='*80}")

    # Print summary table
    print_ga_summary(all_results)


if __name__ == '__main__':
    main()