#!/usr/bin/env python3
"""
Optional Algorithm (Simulated Annealing) experiments for JSSP - Large Dataset (abz7).
"""

import sys
import os
# Get the absolute path of the experiments directory and navigate to project root
experiments_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(experiments_dir))
sys.path.insert(0, project_root)

import traceback

from shared.parser import get_instance_by_name
from shared.sa_algorithm import simulated_annealing
from shared.experiment_utils import (
    setup_experiment, save_results_csv, print_sa_summary, generate_sa_evolution_plots
)


def main():
    """Run SA experiments and save results."""
    # Setup experiment
    results_dir, timestamp, log_file, logger, original_stdout = setup_experiment('optional-large', 'sa')

    print("="*80)
    print(f"Optional Algorithm Experiments Log - Large Dataset (abz7) - {timestamp}")
    print("="*80)

    # Experiment configurations
    configs = [
        {'max_iterations': 5000, 'initial_temperature': 1000, 'cooling_rate': 0.995, 'mutation': 'swap'},
        {'max_iterations': 5000, 'initial_temperature': 1000, 'cooling_rate': 0.995, 'mutation': 'inversion'},
        {'max_iterations': 10000, 'initial_temperature': 1500, 'cooling_rate': 0.99, 'mutation': 'swap'},
        {'max_iterations': 10000, 'initial_temperature': 1500, 'cooling_rate': 0.99, 'mutation': 'inversion'},
        {'max_iterations': 15000, 'initial_temperature': 2000, 'cooling_rate': 0.995, 'mutation': 'swap'},
        {'max_iterations': 15000, 'initial_temperature': 2000, 'cooling_rate': 0.99, 'mutation': 'inversion'},
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

    # Restore stdout
    sys.stdout = original_stdout

    # Save summary CSV
    summary_file = save_results_csv(all_results, results_dir, timestamp, exclude_fields=['history'])

    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Log: {log_file}")
    print(f"Summary: {summary_file}")
    print(f"{'='*80}")

    # Print summary table
    print_sa_summary(all_results)

    # Generate evolution plots
    print(f"\n{'='*80}")
    print(f"Generating evolution plots...")
    print(f"{'='*80}")
    generate_sa_evolution_plots(all_results, results_dir, 'abz7', timestamp)


if __name__ == '__main__':
    main()