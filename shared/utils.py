"""
Utility functions for logging, result saving, and visualization.

This module provides functions to:
- Save experiment results to CSV
- Generate evolution plots
- Create Gantt charts for schedules
"""

import os
import csv
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from datetime import datetime


def save_results_to_csv(results: List[Dict], filepath: str):
    """
    Save experiment results to a CSV file.

    Args:
        results: List of dictionaries with experiment results
        filepath: Path to output CSV file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def plot_evolution(best_makespan_history: List[int], avg_makespan_history: List[int],
                    title: str = "Evolution of Makespan", output_path: str = None):
    """
    Generate and save evolution plot of makespan over generations.

    Args:
        best_makespan_history: List of best makespan per generation
        avg_makespan_history: List of average makespan per generation
        title: Plot title
        output_path: Path to save the plot (if None, displays instead)
    """
    generations = range(len(best_makespan_history))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_makespan_history, 'b-', label='Best Makespan', linewidth=2)
    plt.plot(generations, avg_makespan_history, 'r--', label='Average Makespan', alpha=0.7)

    plt.xlabel('Generation')
    plt.ylabel('Makespan')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_gantt_chart(schedule: List[Tuple[int, int, int, int, int]],
                     instance_name: str, output_path: str = None):
    """
    Generate and save Gantt chart for a schedule.

    Args:
        schedule: List of (job_id, task_id, machine_id, start_time, completion_time)
        instance_name: Name of the JSSP instance
        output_path: Path to save the plot (if None, displays instead)
    """
    # Group tasks by machine
    machine_tasks = {}
    for job_id, task_id, machine_id, start_time, completion_time in schedule:
        if machine_id not in machine_tasks:
            machine_tasks[machine_id] = []
        machine_tasks[machine_id].append((start_time, completion_time, job_id, task_id))

    # Sort machines
    sorted_machines = sorted(machine_tasks.keys())

    # Create color map for jobs
    num_jobs = max([task[2] for tasks in machine_tasks.values() for task in tasks]) + 1
    colors = plt.cm.tab20(range(num_jobs))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, machine_id in enumerate(sorted_machines):
        tasks = machine_tasks[machine_id]
        for start, end, job_id, task_id in tasks:
            ax.barh(i, end - start, left=start, height=0.6,
                    color=colors[job_id], edgecolor='black', linewidth=0.5)
            ax.text((start + end) / 2, i, f'J{job_id}T{task_id}',
                    ha='center', va='center', fontsize=8, color='white', weight='bold')

    ax.set_yticks(range(len(sorted_machines)))
    ax.set_yticklabels([f'Machine {m}' for m in sorted_machines])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title(f'Gantt Chart - {instance_name}')
    ax.grid(True, axis='x', alpha=0.3)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_experiment_summary(instance_name: str, parameters: Dict,
                              best_makespan: int, best_solution: List[int],
                              generations: int, population_size: int,
                              output_path: str = None):
    """
    Create a summary text file for an experiment.

    Args:
        instance_name: Name of the JSSP instance
        parameters: Dictionary of algorithm parameters
        best_makespan: Best makespan found
        best_solution: Best chromosome found
        generations: Number of generations run
        population_size: Population size used
        output_path: Path to save the summary (if None, returns string)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = f"""
{'=' * 60}
JSSP Genetic Algorithm - Experiment Summary
{'=' * 60}

Instance: {instance_name}
Timestamp: {timestamp}

{'=' * 60}
Parameters
{'=' * 60}
"""
    for key, value in parameters.items():
        summary += f"{key}: {value}\n"

    summary += f"""
{'=' * 60}
Results
{'=' * 60}
Best Makespan: {best_makespan}
Generations: {generations}
Population Size: {population_size}

Best Chromosome:
{best_solution}

{'=' * 60}
"""

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(summary)
    else:
        return summary


def get_timestamp() -> str:
    """
    Get current timestamp as a string suitable for filenames.

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%MSS")
