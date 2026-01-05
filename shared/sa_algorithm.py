#!/usr/bin/env python3
"""
Shared Simulated Annealing implementation.
"""

import sys
import os
import random
import math
import importlib.util

from shared.encoding import create_random_chromosome
from shared.fitness import compute_makespan

# Load operators
project_root = os.path.dirname(os.path.dirname(os.getcwd()))
spec = importlib.util.spec_from_file_location('operators', os.path.join(project_root, 'genetic-algorithm', 'operators.py'))
operators = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operators)


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