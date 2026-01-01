"""
Fitness function for JSSP - calculates makespan.

The makespan is the total time required to complete all tasks across all machines.
This module computes the makespan while respecting:
1. Job sequence constraints (tasks must be completed in order)
2. Machine occupancy constraints (one task at a time per machine)
"""

from typing import List, Tuple
from shared.models import JSSPInstance


def compute_makespan(instance: JSSPInstance, chromosome: List[int]) -> int:
    """
    Compute the makespan for a given chromosome.

    Chromosome encoding: permutation with repetitions
    Each element represents a job ID, and the k-th occurrence of job j
    means the k-th task of job j.

    Args:
        instance: JSSPInstance with problem data
        chromosome: List of job IDs representing task order

    Returns:
        Makespan (total completion time)

    Example:
        For 3 jobs with 2 tasks each, chromosome [0, 1, 2, 0, 1, 2] means:
        - Task 0 of Job 0, Task 0 of Job 1, Task 0 of Job 2
        - Task 1 of Job 0, Task 1 of Job 1, Task 1 of Job 2
    """
    num_jobs = instance.num_jobs
    num_machines = instance.num_machines

    # Track completion time for each task in each job
    # job_task_completion[job_id][task_id] = completion_time
    job_task_completion = [[-1] * len(instance.jobs[j].tasks) for j in range(num_jobs)]

    # Track when each machine becomes available
    machine_available = [0] * num_machines

    # Track next task to be processed for each job
    next_task = [0] * num_jobs

    # Process tasks in chromosome order
    for job_id in chromosome:
        task_idx = next_task[job_id]

        # Get task details
        job = instance.get_job(job_id)
        task = job.get_task(task_idx)

        # Calculate earliest start time
        # 1. Must wait for previous task in same job to complete
        job_prev_completion = 0
        if task_idx > 0:
            job_prev_completion = job_task_completion[job_id][task_idx - 1]

        # 2. Must wait for machine to become available
        machine_ready_time = machine_available[task.machine]

        # Start time is the maximum of both constraints
        start_time = max(job_prev_completion, machine_ready_time)

        # Completion time
        completion_time = start_time + task.processing_time

        # Update tracking
        job_task_completion[job_id][task_idx] = completion_time
        machine_available[task.machine] = completion_time
        next_task[job_id] += 1

    # Makespan is the maximum completion time across all jobs
    makespan = 0
    for j in range(num_jobs):
        job_last_completion = job_task_completion[j][-1]
        makespan = max(makespan, job_last_completion)

    return makespan


def compute_detailed_schedule(instance: JSSPInstance, chromosome: List[int]) -> List[Tuple[int, int, int, int, int]]:
    """
    Compute detailed schedule for a given chromosome.

    Args:
        instance: JSSPInstance with problem data
        chromosome: List of job IDs representing task order

    Returns:
        List of tuples (job_id, task_id, machine_id, start_time, completion_time)
    """
    num_jobs = instance.num_jobs
    num_machines = instance.num_machines

    job_task_completion = [[-1] * len(instance.jobs[j].tasks) for j in range(num_jobs)]
    machine_available = [0] * num_machines
    next_task = [0] * num_jobs

    schedule = []

    for job_id in chromosome:
        task_idx = next_task[job_id]

        job = instance.get_job(job_id)
        task = job.get_task(task_idx)

        job_prev_completion = 0
        if task_idx > 0:
            job_prev_completion = job_task_completion[job_id][task_idx - 1]

        machine_ready_time = machine_available[task.machine]
        start_time = max(job_prev_completion, machine_ready_time)
        completion_time = start_time + task.processing_time

        schedule.append((job_id, task_idx, task.machine, start_time, completion_time))

        job_task_completion[job_id][task_idx] = completion_time
        machine_available[task.machine] = completion_time
        next_task[job_id] += 1

    return schedule


def validate_chromosome(instance: JSSPInstance, chromosome: List[int]) -> bool:
    """
    Validate that a chromosome is a valid encoding.

    Args:
        instance: JSSPInstance with problem data
        chromosome: List of job IDs

    Returns:
        True if valid, False otherwise
    """
    expected_length = sum(len(job.tasks) for job in instance.jobs)

    if len(chromosome) != expected_length:
        return False

    # Count occurrences of each job
    job_counts = [0] * instance.num_jobs
    for job_id in chromosome:
        if job_id < 0 or job_id >= instance.num_jobs:
            return False
        job_counts[job_id] += 1

    # Each job should appear exactly as many times as it has tasks
    for j in range(instance.num_jobs):
        if job_counts[j] != len(instance.jobs[j].tasks):
            return False

    return True