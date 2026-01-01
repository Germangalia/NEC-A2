"""
Chromosome encoding and initialization for JSSP Genetic Algorithm.

Encoding: Permutation with Repetitions
- Each job ID appears in the chromosome as many times as it has tasks
- The k-th occurrence of job j represents the k-th task of job j
- The order of job IDs in the chromosome determines the task execution order

Example:
For 3 jobs with tasks:
- Job 0: 3 tasks
- Job 1: 2 tasks
- Job 2: 2 tasks

Valid chromosome: [0, 1, 2, 0, 1, 0, 2]
- First 0 -> Task 0 of Job 0
- First 1 -> Task 0 of Job 1
- First 2 -> Task 0 of Job 2
- Second 0 -> Task 1 of Job 0
- Second 1 -> Task 1 of Job 1
- Third 0 -> Task 2 of Job 0
- Second 2 -> Task 1 of Job 2
"""

import random
from typing import List
from shared.models import JSSPInstance


def create_random_chromosome(instance: JSSPInstance) -> List[int]:
    """
    Create a random valid chromosome for the given JSSP instance.

    Args:
        instance: JSSPInstance with problem data

    Returns:
        Random valid chromosome (list of job IDs)
    """
    chromosome = []

    # Add each job ID as many times as it has tasks
    for job in instance.jobs:
        chromosome.extend([job.job_id] * len(job.tasks))

    # Shuffle to create random order
    random.shuffle(chromosome)

    return chromosome


def initialize_population(instance: JSSPInstance, population_size: int) -> List[List[int]]:
    """
    Initialize a population of random chromosomes.

    Args:
        instance: JSSPInstance with problem data
        population_size: Number of chromosomes in the population

    Returns:
        List of chromosomes
    """
    population = []
    for _ in range(population_size):
        chromosome = create_random_chromosome(instance)
        population.append(chromosome)
    return population


def decode_chromosome(instance: JSSPInstance, chromosome: List[int]) -> List[tuple]:
    """
    Decode a chromosome into a sequence of (job_id, task_id) pairs.

    Args:
        instance: JSSPInstance with problem data
        chromosome: List of job IDs

    Returns:
        List of (job_id, task_id) tuples representing task execution order
    """
    task_counter = [0] * instance.num_jobs
    decoded = []

    for job_id in chromosome:
        task_id = task_counter[job_id]
        decoded.append((job_id, task_id))
        task_counter[job_id] += 1

    return decoded


def encode_task_sequence(instance: JSSPInstance, task_sequence: List[tuple]) -> List[int]:
    """
    Encode a task sequence into a chromosome.

    Args:
        instance: JSSPInstance with problem data
        task_sequence: List of (job_id, task_id) tuples

    Returns:
        Chromosome (list of job IDs)
    """
    chromosome = [job_id for job_id, _ in task_sequence]
    return chromosome


def get_chromosome_length(instance: JSSPInstance) -> int:
    """
    Get the required chromosome length for a JSSP instance.

    Args:
        instance: JSSPInstance with problem data

    Returns:
        Chromosome length (sum of all tasks)
    """
    return sum(len(job.tasks) for job in instance.jobs)


def clone_chromosome(chromosome: List[int]) -> List[int]:
    """
    Create a deep copy of a chromosome.

    Args:
        chromosome: Original chromosome

    Returns:
        Copy of the chromosome
    """
    return chromosome.copy()


def is_valid_chromosome(instance: JSSPInstance, chromosome: List[int]) -> bool:
    """
    Check if a chromosome is valid for the given instance.

    Args:
        instance: JSSPInstance with problem data
        chromosome: Chromosome to validate

    Returns:
        True if valid, False otherwise
    """
    from shared.validator import validate_chromosome
    is_valid, _ = validate_chromosome(instance, chromosome)
    return is_valid