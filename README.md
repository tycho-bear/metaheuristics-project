# Metaheuristics Project: Eagle Strategy with Levy Flights and local search optimization

## Overview
The project combines two well-known metaheuristics for optimization:

1. **Lévy Flight for Global Search**: A stochastic search strategy used for global exploration. It follows a Lévy distribution to create a biased random walk.

2. **Differential Evolution for Local Search**: Once a promising region has been found, differential evolution is used as a local search method to refine solutions in that region. DE is applied iteratively, adjusting the population of agents with mutation, crossover, and selection strategies.

3. **Firefly Algorithm for Local Search**: This serves the same purpose as DE. It is used as a local search to refine solutions.

The **Eagle Strategy (ES)** first uses Lévy flights to explore the search space and then switches to a local search to fine-tune solutions in promising regions.

## Requirements
The following libraries are required to run the code:
- `numpy`
- `matplotlib`

To install them, run:
```bash
pip install -r requirments.txt
```

## Usage

This project can be run from the command line, or with Docker.

### Command Line

To run the code, execute the following command in your terminal:

```bash
python EagleStrategy.py
```

By default, this will run ES + DE on the Ackley function. To run other configurations, you can specify the algorithm and function like this:

```bash
python EagleStrategy.py <ALGORITHM> <FUNCTION_TO_OPTIMIZE>
```

where 

`<ALGORITHM>` = `de` or `firefly`

`<FUNCTION_TO_OPTIMIZE>` = `ackley`, `sphere`, `rosenbrock`, `schwefel`, or `shubert`

### Docker

The simplest way to run this project with Docker is through `make`. 
