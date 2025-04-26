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

To install the dependencies run:
```bash
pip install -r requirments.txt
```

## Usage
To run the code, execute the following command in your terminal:
```bash
python EagleStrategy.py
```

By default, this will run ES + DE on the Ackley function. To run it on the other functions, comment or uncomment these lines of code accordingly, based on the function and local optimizer you want to use.

#### Function to optimize:

```angular2html
f = ackley
# f = sphere
# f = rosenbrock
# f = schwefel
# f = shubert_single
```

#### Set the bounds for the search space:

```angular2html
global_bounds = [(-32.768, 32.768)] * dim_test  # Ackley
# global_bounds = [(-5.12, 5.12)] * dim_test      # De Jong (sphere)
# global_bounds = [(-5, 5)] * dim_test            # Rosenbrock
# global_bounds = [(-500, 500)] * dim_test        # Schwefel
# global_bounds = [(-10, 10)] * dim_test          # Shubert single
```

#### Choose a local optimizer:

```angular2html
opt = DifferentialEvolution(f, global_bounds, mut_factor, crossover_rate, n_agents=n_agents_num)
# opt = Firefly(f, global_bounds, beta0=beta0, gamma=gamma, alpha=alpha, theta=theta, n_agents=n_agents_num)
```
