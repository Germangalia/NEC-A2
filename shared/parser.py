"""
Parser for JSSP datasets in OR-Library format.

Format:
instance <name>
<description>
<num_jobs> <num_machines>
<machine_0> <time_0> <machine_1> <time_1> ... for job 0
<machine_0> <time_0> <machine_1> <time_1> ... for job 1
...
"""

import re
from typing import List, Dict
from shared.models import Task, Job, JSSPInstance


def parse_jobshop_file(filepath: str) -> Dict[str, JSSPInstance]:
    """
    Parse a jobshop1.txt file containing multiple JSSP instances.

    Args:
        filepath: Path to the jobshop1.txt file

    Returns:
        Dictionary mapping instance names to JSSPInstance objects
    """
    with open(filepath, 'r') as f:
        content = f.read()

    instances = {}
    lines = content.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for instance declaration
        if line.startswith('instance '):
            instance_name = line.split(' ', 1)[1].strip()
            i += 1

            # Skip description line
            i += 1

            # Parse num_jobs and num_machines
            while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith('+')):
                i += 1

            if i >= len(lines):
                break

            header_line = lines[i].strip().split()
            num_jobs = int(header_line[0])
            num_machines = int(header_line[1])
            i += 1

            # Parse jobs
            jobs = []
            for job_id in range(num_jobs):
                if i >= len(lines):
                    break

                job_line = lines[i].strip()
                while not job_line and i < len(lines) - 1:
                    i += 1
                    job_line = lines[i].strip()

                values = list(map(int, job_line.split()))
                tasks = []
                for j in range(0, len(values), 2):
                    if j + 1 < len(values):
                        machine = values[j]
                        processing_time = values[j + 1]
                        tasks.append(Task(machine=machine, processing_time=processing_time))

                jobs.append(Job(job_id=job_id, tasks=tasks))
                i += 1

            instance = JSSPInstance(
                instance_name=instance_name,
                num_jobs=num_jobs,
                num_machines=num_machines,
                jobs=jobs
            )
            instances[instance_name] = instance

        i += 1

    return instances


def get_instance_by_name(filepath: str, instance_name: str) -> JSSPInstance:
    """
    Parse jobshop file and return a specific instance by name.

    Args:
        filepath: Path to the jobshop1.txt file
        instance_name: Name of the instance to retrieve

    Returns:
        JSSPInstance object

    Raises:
        ValueError: If instance name is not found
    """
    instances = parse_jobshop_file(filepath)
    if instance_name not in instances:
        available = ', '.join(instances.keys())
        raise ValueError(f"Instance '{instance_name}' not found. Available instances: {available}")
    return instances[instance_name]


def list_instances(filepath: str) -> List[str]:
    """
    List all available instance names in the jobshop file.

    Args:
        filepath: Path to the jobshop1.txt file

    Returns:
        List of instance names
    """
    instances = parse_jobshop_file(filepath)
    return list(instances.keys())