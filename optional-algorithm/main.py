"""
Simulated Annealing implementation for JSSP.

This module implements Simulated Annealing as a second optimization method,
reusing the shared codebase (parser, fitness, models).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import math
from typing import List, Tuple, Callable

from shared.models import JSSPInstance
from shared.fitness import compute_makespan
from shared.encoding import create_random_chromosome
from genetic_algorithm.operators import swap_mutation, inversion_mutation


class SimulatedAnnealing:
    """Simulated Annealing for Job Shop Scheduling Problem."""

    def __init__(self,
                 instance: JSSPInstance,
                 initial_temperature: float = 1000.0,
                 final_temperature: float = 0.1,
                 cooling_rate: float = 0.995,
                 max_iterations: int = 10000,
                 mutation_method: str = 'swap',
                 random_seed: int = None):
        """
        Initialize SA parameters.

        Args:
            instance: JSSPInstance to solve
            initial_temperature: Starting temperature
            final_temperature: Minimum temperature to stop
            cooling_rate: Temperature reduction factor (0 < rate < 1)
            max_iterations: Maximum number of iterations
            mutation_method: 'swap' or 'inversion' for neighbor generation
            random_seed: Random seed for reproducibility
        """
        self.instance = instance
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.mutation_method = mutation_method

        if random_seed is not None:
            random.seed(random_seed)

        # History tracking
        self.best_makespan_history = []
        self.best_chromosome = None
        self.best_makespan = float('inf')

    def generate_neighbor(self, chromosome: List[int]) -> List[int]:
        """
        Generate a neighbor solution using mutation operators.

        Args:
            chromosome: Current solution

        Returns:
            Neighbor solution
        """
        if self.mutation_method == 'swap':
            return swap_mutation(chromosome)
        elif self.mutation_method == 'inversion':
            return inversion_mutation(chromosome)
        else:
            raise ValueError(f"Unknown mutation method: {self.mutation_method}")

    def acceptance_probability(self, current_cost: int, new_cost: int, temperature: float) -> float:
        """
        Calculate acceptance probability for a worse solution.

        Args:
            current_cost: Cost of current solution
            new_cost: Cost of new solution
            temperature: Current temperature

        Returns:
            Acceptance probability (0 to 1)
        """
        if new_cost < current_cost:
            return 1.0
        else:
            return math.exp((current_cost - new_cost) / temperature)

    def run(self) -> Tuple[List[int], int]:
        """
        Run the Simulated Annealing algorithm.

        Returns:
            Tuple of (best_chromosome, best_makespan)
        """
        # Initialize with random solution
        current_solution = create_random_chromosome(self.instance)
        current_cost = compute_makespan(self.instance, current_solution)

        # Initialize best solution
        best_solution = current_solution.copy()
        best_cost = current_cost

        self.best_chromosome = best_solution
        self.best_makespan = best_cost

        temperature = self.initial_temperature

        for iteration in range(self.max_iterations):
            # Track history
            self.best_makespan_history.append(best_cost)

            # Generate neighbor
            neighbor = self.generate_neighbor(current_solution)
            neighbor_cost = compute_makespan(self.instance, neighbor)

            # Accept or reject neighbor
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_solution = neighbor
                current_cost = neighbor_cost

                # Update best solution
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
                    self.best_chromosome = best_solution
                    self.best_makespan = best_cost

            # Cool down
            temperature *= self.cooling_rate

            # Check termination
            if temperature < self.final_temperature:
                print(f"Final temperature reached at iteration {iteration}")
                break

            # Print progress
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}: Temp = {temperature:.2f}, Best = {best_cost}")

        print(f"Final: Best makespan = {best_cost}")
        return best_solution, best_cost


def main():
    """Main function with command-line argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Simulated Annealing for Job Shop Scheduling Problem',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset file (jobshop1.txt)')
    parser.add_argument('--instance', type=str, required=True,
                        help='Instance name to solve')

    # SA parameters
    parser.add_argument('--initial-temperature', type=float, default=1000.0,
                        help='Initial temperature')
    parser.add_argument('--final-temperature', type=float, default=0.1,
                        help='Final temperature')
    parser.add_argument('--cooling-rate', type=float, default=0.995,
                        help='Cooling rate (temperature multiplier)')
    parser.add_argument('--max-iterations', type=int, default=10000,
                        help='Maximum number of iterations')
    parser.add_argument('--mutation', type=str, default='swap',
                        choices=['swap', 'inversion'],
                        help='Mutation method for neighbor generation')

    # Other
    parser.add_argument('--random-seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='../results',
                        help='Output directory for results')

    args = parser.parse_args()

    # Load instance
    from shared.parser import get_instance_by_name
    from shared.utils import (
        save_results_to_csv, plot_evolution, plot_gantt_chart,
        create_experiment_summary, get_timestamp
    )
    from shared.fitness import compute_detailed_schedule

    print(f"Loading instance '{args.instance}' from {args.dataset}")
    instance = get_instance_by_name(args.dataset, args.instance)
    print(f"Instance: {instance.num_jobs} jobs, {instance.num_machines} machines")

    # Create and run SA
    print("\nRunning Simulated Annealing...")
    sa = SimulatedAnnealing(
        instance=instance,
        initial_temperature=args.initial_temperature,
        final_temperature=args.final_temperature,
        cooling_rate=args.cooling_rate,
        max_iterations=args.max_iterations,
        mutation_method=args.mutation,
        random_seed=args.random_seed
    )

    best_chromosome, best_makespan = sa.run()

    # Generate outputs
    timestamp = get_timestamp()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save evolution plot
    evolution_path = os.path.join(output_dir, f'{args.instance}_sa_evolution_{timestamp}.png')
    # Use empty list for avg makespan since SA doesn't track it
    plot_evolution(sa.best_makespan_history, [],
                   title=f'SA Evolution - {args.instance}', output_path=evolution_path)
    print(f"Evolution plot saved to {evolution_path}")

    # Generate Gantt chart
    schedule = compute_detailed_schedule(instance, best_chromosome)
    gantt_path = os.path.join(output_dir, f'{args.instance}_sa_gantt_{timestamp}.png')
    plot_gantt_chart(schedule, args.instance, output_path=gantt_path)
    print(f"Gantt chart saved to {gantt_path}")

    # Save experiment summary
    summary_path = os.path.join(output_dir, f'{args.instance}_sa_summary_{timestamp}.txt')
    parameters = {
        'initial_temperature': args.initial_temperature,
        'final_temperature': args.final_temperature,
        'cooling_rate': args.cooling_rate,
        'max_iterations': args.max_iterations,
        'mutation_method': args.mutation,
        'random_seed': args.random_seed
    }
    create_experiment_summary(args.instance, parameters, best_makespan, best_chromosome,
                            len(sa.best_makespan_history), 1, summary_path)
    print(f"Experiment summary saved to {summary_path}")

    # Save results to CSV
    results = [{
        'algorithm': 'Simulated Annealing',
        'instance': args.instance,
        'best_makespan': best_makespan,
        'iterations': len(sa.best_makespan_history),
        'initial_temperature': args.initial_temperature,
        'final_temperature': args.final_temperature,
        'cooling_rate': args.cooling_rate,
        'mutation_method': args.mutation,
        'random_seed': args.random_seed,
        'timestamp': timestamp
    }]
    csv_path = os.path.join(output_dir, f'{args.instance}_sa_results_{timestamp}.csv')
    save_results_to_csv(results, csv_path)
    print(f"Results saved to {csv_path}")

    print(f"\n{'='*60}")
    print(f"Best makespan: {best_makespan}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()