"""
Genetic Algorithm main implementation for JSSP.

This module implements the complete GA with:
- Population initialization
- Fitness evaluation
- Selection
- Crossover
- Mutation
- Elitism
- Stationary state detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Callable
import random

from shared.models import JSSPInstance
from shared.fitness import compute_makespan
from shared.encoding import initialize_population
from operators import (
    tournament_selection, select_parents_tournament,
    rank_selection, select_parents_rank,
    crossover_ox, crossover_pmx,
    mutate_swap, mutate_inversion
)


class GeneticAlgorithm:
    """Genetic Algorithm for Job Shop Scheduling Problem."""

    def __init__(self,
                 instance: JSSPInstance,
                 population_size: int = 100,
                 max_generations: int = 500,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elite_size: int = 2,
                 selection_method: str = 'tournament',
                 crossover_method: str = 'ox',
                 mutation_method: str = 'swap',
                 tournament_size: int = 3,
                 stationary_generations: int = 50,
                 random_seed: int = None):
        """
        Initialize GA parameters.

        Args:
            instance: JSSPInstance to solve
            population_size: Size of population
            max_generations: Maximum number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elite_size: Number of elite individuals to preserve
            selection_method: 'tournament' or 'rank'
            crossover_method: 'ox' or 'pmx'
            mutation_method: 'swap' or 'inversion'
            tournament_size: Size for tournament selection
            stationary_generations: Generations without improvement to stop
            random_seed: Random seed for reproducibility
        """
        self.instance = instance
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.tournament_size = tournament_size
        self.stationary_generations = stationary_generations

        if random_seed is not None:
            random.seed(random_seed)

        # History tracking
        self.best_makespan_history = []
        self.avg_makespan_history = []
        self.best_chromosome = None
        self.best_makespan = float('inf')

    def evaluate_population(self, population: List[List[int]]) -> List[int]:
        """
        Evaluate fitness of all chromosomes in population.

        Args:
            population: List of chromosomes

        Returns:
            List of fitness values (makespan) for each chromosome
        """
        fitness_values = []
        for chromosome in population:
            makespan = compute_makespan(self.instance, chromosome)
            fitness_values.append(makespan)
        return fitness_values

    def select_parents(self, population: List[List[int]], fitness_values: List[int]) -> List[List[int]]:
        """
        Select parents for reproduction.

        Args:
            population: Current population
            fitness_values: Fitness values for population

        Returns:
            List of selected parent chromosomes
        """
        num_parents = self.population_size - self.elite_size

        if self.selection_method == 'tournament':
            return select_parents_tournament(population, fitness_values, num_parents, self.tournament_size)
        elif self.selection_method == 'rank':
            return select_parents_rank(population, fitness_values, num_parents)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

    def apply_crossover(self, parents: List[List[int]]) -> List[List[int]]:
        """
        Apply crossover to parents.

        Args:
            parents: List of parent chromosomes

        Returns:
            List of offspring chromosomes
        """
        if self.crossover_method == 'ox':
            return crossover_ox(parents, self.crossover_rate)
        elif self.crossover_method == 'pmx':
            return crossover_pmx(parents, self.crossover_rate)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")

    def apply_mutation(self, population: List[List[int]]) -> List[List[int]]:
        """
        Apply mutation to population.

        Args:
            population: List of chromosomes

        Returns:
            List of mutated chromosomes
        """
        if self.mutation_method == 'swap':
            return mutate_swap(population, self.mutation_rate)
        elif self.mutation_method == 'inversion':
            return mutate_inversion(population, self.mutation_rate)
        else:
            raise ValueError(f"Unknown mutation method: {self.mutation_method}")

    def apply_elitism(self, population: List[List[int]], fitness_values: List[int]) -> Tuple[List[List[int]], List[int]]:
        """
        Preserve elite individuals for next generation.

        Args:
            population: Current population
            fitness_values: Fitness values for population

        Returns:
            Tuple of (elite_chromosomes, elite_fitness_values)
        """
        # Sort by fitness (ascending - lower is better)
        sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])

        # Get elite individuals
        elite_indices = sorted_indices[:self.elite_size]
        elite_chromosomes = [population[i].copy() for i in elite_indices]
        elite_fitness = [fitness_values[i] for i in elite_indices]

        return elite_chromosomes, elite_fitness

    def check_stationary_state(self) -> bool:
        """
        Check if the algorithm has reached a stationary state.

        Returns:
            True if no improvement in last stationary_generations
        """
        if len(self.best_makespan_history) < self.stationary_generations:
            return False

        recent_best = self.best_makespan_history[-self.stationary_generations:]
        return all(best == recent_best[0] for best in recent_best)

    def run(self) -> Tuple[List[int], int]:
        """
        Run the genetic algorithm.

        Returns:
            Tuple of (best_chromosome, best_makespan)
        """
        # Initialize population
        population = initialize_population(self.instance, self.population_size)

        for generation in range(self.max_generations):
            # Evaluate population
            fitness_values = self.evaluate_population(population)

            # Track best solution
            min_fitness = min(fitness_values)
            avg_fitness = sum(fitness_values) / len(fitness_values)

            self.best_makespan_history.append(min_fitness)
            self.avg_makespan_history.append(avg_fitness)

            # Update global best
            if min_fitness < self.best_makespan:
                self.best_makespan = min_fitness
                best_idx = fitness_values.index(min_fitness)
                self.best_chromosome = population[best_idx].copy()

            # Check stationary state
            if self.check_stationary_state():
                print(f"Stationary state reached at generation {generation}")
                break

            # Selection
            parents = self.select_parents(population, fitness_values)

            # Crossover
            offspring = self.apply_crossover(parents)

            # Mutation
            offspring = self.apply_mutation(offspring)

            # Elitism: preserve best individuals
            elite_chromosomes, elite_fitness = self.apply_elitism(population, fitness_values)

            # Create new population
            population = offspring + elite_chromosomes

            if generation % 50 == 0:
                print(f"Generation {generation}: Best = {min_fitness}, Avg = {avg_fitness:.2f}")

        print(f"Final: Best makespan = {self.best_makespan}")
        return self.best_chromosome, self.best_makespan


def main():
    """Main function with command-line argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Genetic Algorithm for Job Shop Scheduling Problem',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset file (jobshop1.txt)')
    parser.add_argument('--instance', type=str, required=True,
                        help='Instance name to solve')

    # GA parameters
    parser.add_argument('--population-size', type=int, default=100,
                        help='Population size')
    parser.add_argument('--max-generations', type=int, default=500,
                        help='Maximum number of generations')
    parser.add_argument('--crossover-rate', type=float, default=0.8,
                        help='Crossover probability')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                        help='Mutation probability')
    parser.add_argument('--elite-size', type=int, default=2,
                        help='Number of elite individuals')

    # Operator selection
    parser.add_argument('--selection', type=str, default='tournament',
                        choices=['tournament', 'rank'],
                        help='Selection method')
    parser.add_argument('--crossover', type=str, default='ox',
                        choices=['ox', 'pmx'],
                        help='Crossover method')
    parser.add_argument('--mutation', type=str, default='swap',
                        choices=['swap', 'inversion'],
                        help='Mutation method')
    parser.add_argument('--tournament-size', type=int, default=3,
                        help='Tournament size for tournament selection')

    # Termination
    parser.add_argument('--stationary-generations', type=int, default=50,
                        help='Generations without improvement to stop')

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

    # Create and run GA
    print("\nRunning Genetic Algorithm...")
    ga = GeneticAlgorithm(
        instance=instance,
        population_size=args.population_size,
        max_generations=args.max_generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
        selection_method=args.selection,
        crossover_method=args.crossover,
        mutation_method=args.mutation,
        tournament_size=args.tournament_size,
        stationary_generations=args.stationary_generations,
        random_seed=args.random_seed
    )

    best_chromosome, best_makespan = ga.run()

    # Generate outputs
    timestamp = get_timestamp()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save evolution plot
    evolution_path = os.path.join(output_dir, f'{args.instance}_evolution_{timestamp}.png')
    plot_evolution(ga.best_makespan_history, ga.avg_makespan_history,
                   title=f'Evolution - {args.instance}', output_path=evolution_path)
    print(f"Evolution plot saved to {evolution_path}")

    # Generate Gantt chart
    schedule = compute_detailed_schedule(instance, best_chromosome)
    gantt_path = os.path.join(output_dir, f'{args.instance}_gantt_{timestamp}.png')
    plot_gantt_chart(schedule, args.instance, output_path=gantt_path)
    print(f"Gantt chart saved to {gantt_path}")

    # Save experiment summary
    summary_path = os.path.join(output_dir, f'{args.instance}_summary_{timestamp}.txt')
    parameters = {
        'population_size': args.population_size,
        'max_generations': args.max_generations,
        'crossover_rate': args.crossover_rate,
        'mutation_rate': args.mutation_rate,
        'elite_size': args.elite_size,
        'selection_method': args.selection,
        'crossover_method': args.crossover,
        'mutation_method': args.mutation,
        'tournament_size': args.tournament_size,
        'stationary_generations': args.stationary_generations,
        'random_seed': args.random_seed
    }
    create_experiment_summary(args.instance, parameters, best_makespan, best_chromosome,
                            len(ga.best_makespan_history), args.population_size, summary_path)
    print(f"Experiment summary saved to {summary_path}")

    # Save results to CSV
    results = [{
        'instance': args.instance,
        'best_makespan': best_makespan,
        'generations': len(ga.best_makespan_history),
        'population_size': args.population_size,
        'selection_method': args.selection,
        'crossover_method': args.crossover,
        'mutation_method': args.mutation,
        'crossover_rate': args.crossover_rate,
        'mutation_rate': args.mutation_rate,
        'elite_size': args.elite_size,
        'timestamp': timestamp
    }]
    csv_path = os.path.join(output_dir, f'{args.instance}_results_{timestamp}.csv')
    save_results_to_csv(results, csv_path)
    print(f"Results saved to {csv_path}")

    print(f"\n{'='*60}")
    print(f"Best makespan: {best_makespan}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()