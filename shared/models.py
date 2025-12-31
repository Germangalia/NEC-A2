"""
Data structures for Job Shop Scheduling Problem (JSSP).

Each task is defined as a pair (m, p) where:
- m: machine ID (0-indexed)
- p: processing time
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Task:
    """Represents a single task in a job."""
    machine: int
    processing_time: int

    def __repr__(self) -> str:
        return f"Task(m={self.machine}, p={self.processing_time})"


@dataclass
class Job:
    """Represents a job consisting of a sequence of tasks."""
    job_id: int
    tasks: List[Task]

    def __repr__(self) -> str:
        return f"Job(id={self.job_id}, tasks={len(self.tasks)})"

    def get_num_tasks(self) -> int:
        """Return the number of tasks in this job."""
        return len(self.tasks)

    def get_task(self, index: int) -> Task:
        """Get task at specific index."""
        return self.tasks[index]


@dataclass
class JSSPInstance:
    """Represents a complete JSSP instance."""
    instance_name: str
    num_jobs: int
    num_machines: int
    jobs: List[Job]

    def __repr__(self) -> str:
        return f"JSSPInstance(name={self.instance_name}, jobs={self.num_jobs}, machines={self.num_machines})"

    def get_job(self, job_id: int) -> Job:
        """Get job by ID."""
        return self.jobs[job_id]

    def get_total_tasks(self) -> int:
        """Return total number of tasks across all jobs."""
        return sum(job.get_num_tasks() for job in self.jobs)