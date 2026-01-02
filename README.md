# NEC-A2

Neural and Evolutionary Computation (NEC) - 2025/26. Activity 2 - Optimization with Genetic Algorithms.

## Overview

This project implements optimization algorithms for the Job Shop Scheduling Problem (JSSP):
- **Genetic Algorithm (GA)**: Primary optimization method with multiple operators
- **Simulated Annealing (SA)**: Optional secondary optimization method

## Project Structure

```
NEC-A2/
├── dataset/
│   └── jobshop1.txt          # OR-Library JSSP instances
├── shared/                    # Reusable modules
│   ├── models.py            # Data structures (Task, Job, JSSPInstance)
│   ├── parser.py            # Dataset parser
│   ├── fitness.py           # Makespan calculation and validation
│   ├── validator.py         # Chromosome validation
│   ├── utils.py             # Visualization and logging utilities
│   └── encoding.py          # Chromosome encoding and initialization
├── genetic-algorithm/        # GA implementation
│   ├── operators.py         # Selection, crossover, mutation operators
│   └── main.py              # GA main execution script
├── optional-algorithm/       # SA implementation
│   └── main.py              # SA main execution script
└── results/                  # Experiment outputs
```

## Dataset Selection

Selected JSSP instances from OR-Library:

### Small Dataset (6x6)
- **ft06**: Fisher and Thompson 6x6 instance
  - Jobs: 6, Machines: 6
  - Source: Fisher & Thompson (1963)

### Medium Dataset (10x10)
- **ft10**: Fisher and Thompson 10x10 instance
  - Jobs: 10, Machines: 10
  - Source: Fisher & Thompson (1963)

### Large Dataset (20x15)
- **abz7**: Adams, Balas, and Zawack 20x15 instance
  - Jobs: 20, Machines: 15
  - Source: Adams, Balas & Zawack (1988)

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Genetic Algorithm

```bash
# Run GA with default parameters
python genetic-algorithm/main.py --dataset dataset/jobshop1.txt --instance ft06

# Run GA with custom parameters
python genetic-algorithm/main.py \
    --dataset dataset/jobshop1.txt \
    --instance ft10 \
    --population-size 200 \
    --max-generations 1000 \
    --selection tournament \
    --crossover ox \
    --mutation swap \
    --crossover-rate 0.9 \
    --mutation-rate 0.1 \
    --elite-size 5
```

### Simulated Annealing

```bash
# Run SA with default parameters
python optional-algorithm/main.py --dataset dataset/jobshop1.txt --instance ft06

# Run SA with custom parameters
python optional-algorithm/main.py \
    --dataset dataset/jobshop1.txt \
    --instance abz7 \
    --initial-temperature 1000 \
    --final-temperature 0.1 \
    --cooling-rate 0.995 \
    --max-iterations 10000 \
    --mutation swap
```

## GA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--population-size` | 100 | Population size |
| `--max-generations` | 500 | Maximum generations |
| `--crossover-rate` | 0.8 | Crossover probability |
| `--mutation-rate` | 0.1 | Mutation probability |
| `--elite-size` | 2 | Number of elite individuals |
| `--selection` | tournament | Selection method (tournament/rank) |
| `--crossover` | ox | Crossover method (ox/pmx) |
| `--mutation` | swap | Mutation method (swap/inversion) |
| `--tournament-size` | 3 | Tournament size |
| `--stationary-generations` | 50 | Generations without improvement to stop |

## SA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--initial-temperature` | 1000.0 | Starting temperature |
| `--final-temperature` | 0.1 | Minimum temperature |
| `--cooling-rate` | 0.995 | Temperature reduction factor |
| `--max-iterations` | 10000 | Maximum iterations |
| `--mutation` | swap | Neighbor generation method |

## Output Files

Each experiment generates:
- **Evolution plot**: `{instance}_evolution_{timestamp}.png`
- **Gantt chart**: `{instance}_gantt_{timestamp}.png`
- **Summary**: `{instance}_summary_{timestamp}.txt`
- **Results CSV**: `{instance}_results_{timestamp}.csv`

## Genetic Algorithm Operators

### Selection Methods
- **Tournament Selection**: Selects best individual from random tournament
- **Rank Selection**: Selects based on rank probabilities

### Crossover Methods
- **Order Crossover (OX)**: Preserves relative order of elements
- **Partially Mapped Crossover (PMX)**: Preserves absolute positions

### Mutation Methods
- **Swap Mutation**: Swaps two randomly selected elements
- **Inversion Mutation**: Reverses a randomly selected segment

## Chromosome Encoding

**Permutation with Repetitions**:
- Each job ID appears as many times as it has tasks
- The k-th occurrence of job j represents the k-th task of job j
- Example: `[0, 1, 2, 0, 1, 0, 2]` for 3 jobs with varying task counts
