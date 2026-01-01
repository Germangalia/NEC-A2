"""
Genetic operators for JSSP: Selection, Crossover, and Mutation.

This module implements multiple techniques for each operator:
- Selection: Tournament, Rank
- Crossover: Order Crossover (OX), Partially Mapped Crossover (PMX)
- Mutation: Swap, Inversion
"""

import random
from typing import List, Tuple
from shared.models import JSSPInstance
from shared.fitness import compute_makespan


# ==================== SELECTION OPERATORS ====================

def tournament_selection(population: List[List[int]], fitness_values: List[int],
                        tournament_size: int = 3) -> List[int]:
    """
    Tournament selection: select best individual from random tournament.

    Args:
        population: List of chromosomes
        fitness_values: List of fitness values (makespan) for each chromosome
        tournament_size: Number of individuals in each tournament

    Returns:
        Selected chromosome (best from tournament)
    """
    # Randomly select tournament_size individuals
    tournament_indices = random.sample(range(len(population)), tournament_size)

    # Find the best (lowest makespan) in the tournament
    best_idx = min(tournament_indices, key=lambda i: fitness_values[i])

    return population[best_idx].copy()


def select_parents_tournament(population: List[List[int]], fitness_values: List[int],
                              num_parents: int, tournament_size: int = 3) -> List[List[int]]:
    """
    Select multiple parents using tournament selection.

    Args:
        population: List of chromosomes
        fitness_values: List of fitness values for each chromosome
        num_parents: Number of parents to select
        tournament_size: Size of each tournament

    Returns:
        List of selected parent chromosomes
    """
    parents = []
    for _ in range(num_parents):
        parent = tournament_selection(population, fitness_values, tournament_size)
        parents.append(parent)
    return parents


def rank_selection(population: List[List[int]], fitness_values: List[int]) -> List[int]:
    """
    Rank selection: select based on rank probabilities.

    Individuals are ranked by fitness (best = rank 1).
    Selection probability is proportional to rank (linear ranking).

    Args:
        population: List of chromosomes
        fitness_values: List of fitness values (makespan) for each chromosome

    Returns:
        Selected chromosome based on rank probability
    """
    # Sort indices by fitness (ascending - lower makespan is better)
    sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])

    # Assign ranks (1 = best, N = worst)
    ranks = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}

    # Calculate selection probabilities (linear ranking)
    # Probability for rank r: (2 * (N - r + 1)) / (N * (N + 1))
    N = len(population)
    probabilities = [(2 * (N - ranks[i] + 1)) / (N * (N + 1)) for i in range(N)]

    # Select based on probabilities
    selected_idx = random.choices(range(N), weights=probabilities, k=1)[0]

    return population[selected_idx].copy()


def select_parents_rank(population: List[List[int]], fitness_values: List[int],
                        num_parents: int) -> List[List[int]]:
    """
    Select multiple parents using rank selection.

    Args:
        population: List of chromosomes
        fitness_values: List of fitness values for each chromosome
        num_parents: Number of parents to select

    Returns:
        List of selected parent chromosomes
    """
    parents = []
    for _ in range(num_parents):
        parent = rank_selection(population, fitness_values)
        parents.append(parent)
    return parents


# ==================== CROSSOVER OPERATORS ====================

def order_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Order Crossover (OX): preserves relative order of elements.

    1. Select a random segment from parent1
    2. Copy this segment to child1 at the same positions
    3. Fill remaining positions with elements from parent2 in order

    Args:
        parent1: First parent chromosome
        parent2: Second parent chromosome

    Returns:
        Tuple of (child1, child2)
    """
    size = len(parent1)
    if size < 2:
        return parent1.copy(), parent2.copy()

    # Select two random cut points
    start, end = sorted(random.sample(range(size), 2))

    # Initialize children
    child1 = [-1] * size
    child2 = [-1] * size

    # Copy segment from parent1 to child1 and parent2 to child2
    child1[start:end + 1] = parent1[start:end + 1]
    child2[start:end + 1] = parent2[start:end + 1]

    # Fill remaining positions for child1 using parent2
    def fill_child(child, other_parent):
        used = set([x for x in child if x != -1])
        other_ptr = (end + 1) % size

        for i in range(size):
            pos = (end + 1 + i) % size
            if child[pos] == -1:
                while other_parent[other_ptr] in used:
                    other_ptr = (other_ptr + 1) % size
                child[pos] = other_parent[other_ptr]
                used.add(other_parent[other_ptr])
                other_ptr = (other_ptr + 1) % size

    fill_child(child1, parent2)
    fill_child(child2, parent1)

    return child1, child2


def crossover_ox(parents: List[List[int]], crossover_rate: float = 0.8) -> List[List[int]]:
    """
    Apply Order Crossover to a list of parents.

    Args:
        parents: List of parent chromosomes (must be even number)
        crossover_rate: Probability of applying crossover

    Returns:
        List of offspring chromosomes
    """
    offspring = []

    for i in range(0, len(parents), 2):
        if i + 1 >= len(parents):
            offspring.append(parents[i].copy())
            continue

        parent1 = parents[i]
        parent2 = parents[i + 1]

        if random.random() < crossover_rate:
            child1, child2 = order_crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parent1.copy())
            offspring.append(parent2.copy())

    return offspring