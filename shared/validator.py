"""
Chromosome validation utilities for JSSP.

This module provides validation functions to ensure chromosomes
represent valid solutions for the Job Shop Scheduling Problem.
"""

from typing import List, Tuple, Optional
from shared.models import JSSPInstance
from shared.fitness import compute_detailed_schedule


def validate_chromosome(instance: JSSPInstance, chromosome: List[int]) -> Tuple[bool, Optional[str]]:
    """
    Validate that a chromosome is a valid encoding for the JSSP instance.

    Checks:
    1. Correct length (sum of all tasks)
    2. Valid job IDs (within range)
    3. Correct count of each job (matches number of tasks per job)

    Args:
        instance: JSSPInstance with problem data
        chromosome: List of job IDs

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if chromosome is valid
        - error_message: None if valid, otherwise description of error
    """
    # Check chromosome length
    expected_length = sum(len(job.tasks) for job in instance.jobs)
    if len(chromosome) != expected_length:
        return False, f"Invalid chromosome length: expected {expected_length}, got {len(chromosome)}"

    # Count occurrences of each job
    job_counts = [0] * instance.num_jobs
    for job_id in chromosome:
        # Check valid job ID
        if job_id < 0 or job_id >= instance.num_jobs:
            return False, f"Invalid job ID: {job_id} (must be in range [0, {instance.num_jobs - 1}])"
        job_counts[job_id] += 1

    # Check each job appears correct number of times
    for j in range(instance.num_jobs):
        expected_count = len(instance.jobs[j].tasks)
        if job_counts[j] != expected_count:
            return False, f"Job {j}: expected {expected_count} occurrences, got {job_counts[j]}"

    return True, None


def validate_schedule(instance: JSSPInstance, chromosome: List[int]) -> Tuple[bool, Optional[str]]:
    """
    Validate that the schedule produced by a chromosome is valid.

    Checks:
    1. Chromosome is valid encoding
    2. No overlapping tasks on the same machine
    3. Tasks in each job are in correct order

    Args:
        instance: JSSPInstance with problem data
        chromosome: List of job IDs

    Returns:
        Tuple of (is_valid, error_message)
    """
    # First validate chromosome encoding
    is_valid, error_msg = validate_chromosome(instance, chromosome)
    if not is_valid:
        return False, error_msg

    # Get detailed schedule
    schedule = compute_detailed_schedule(instance, chromosome)

    # Check machine constraints (no overlapping tasks)
    machine_tasks = {}  # machine_id -> list of (start, end, job_id, task_id)
    for job_id, task_id, machine_id, start_time, completion_time in schedule:
        if machine_id not in machine_tasks:
            machine_tasks[machine_id] = []

        # Check for overlaps
        for (existing_start, existing_end, existing_job, existing_task) in machine_tasks[machine_id]:
            if not (completion_time <= existing_start or start_time >= existing_end):
                return False, (
                    f"Machine {machine_id}: Overlap detected - "
                    f"Job {job_id} Task {task_id} ({start_time}-{completion_time}) "
                    f"conflicts with Job {existing_job} Task {existing_task} ({existing_start}-{existing_end})"
                )

        machine_tasks[machine_id].append((start_time, completion_time, job_id, task_id))

    # Check job sequence constraints
    job_last_completion = {}  # job_id -> last completion time
    for job_id, task_id, machine_id, start_time, completion_time in schedule:
        if job_id not in job_last_completion:
            job_last_completion[job_id] = -1

        # First task can start at time 0, others must wait
        if task_id > 0:
            if start_time < job_last_completion[job_id]:
                return False, (
                    f"Job {job_id}: Task {task_id} starts at {start_time} "
                    f"before previous task completed at {job_last_completion[job_id]}"
                )

        job_last_completion[job_id] = completion_time

    return True, None


def get_validation_report(instance: JSSPInstance, chromosome: List[int]) -> dict:
    """
    Generate a comprehensive validation report for a chromosome.

    Args:
        instance: JSSPInstance with problem data
        chromosome: List of job IDs

    Returns:
        Dictionary with validation results and statistics
    """
    is_valid, error_msg = validate_chromosome(instance, chromosome)
    schedule_valid, schedule_error = validate_schedule(instance, chromosome)

    report = {
        'chromosome_valid': is_valid,
        'chromosome_error': error_msg,
        'schedule_valid': schedule_valid,
        'schedule_error': schedule_error,
        'overall_valid': is_valid and schedule_valid,
        'chromosome_length': len(chromosome),
        'expected_length': sum(len(job.tasks) for job in instance.jobs),
        'num_jobs': instance.num_jobs,
        'num_machines': instance.num_machines,
    }

    return report