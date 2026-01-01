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