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
    """Main function for running GA."""
    print("Genetic Algorithm for JSSP - Main Module")
    print("Use this module programmatically or extend with command-line arguments.")


if __name__ == "__main__":
    main()