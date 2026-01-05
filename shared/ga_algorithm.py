#!/usr/bin/env python3
"""
Shared Genetic Algorithm implementation.
"""

import sys
import os
import importlib.util

from shared.encoding import create_random_chromosome
from shared.fitness import compute_makespan

# Load operators
project_root = os.path.dirname(os.path.dirname(os.getcwd()))
spec = importlib.util.spec_from_file_location('operators', os.path.join(project_root, 'genetic-algorithm', 'operators.py'))
operators = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operators)

crossover_ox = operators.crossover_ox
crossover_pmx = operators.crossover_pmx
mutate_swap = operators.mutate_swap
mutate_inversion = operators.mutate_inversion
tournament_selection = operators.tournament_selection
rank_selection = operators.rank_selection


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