import numpy as np
import math
import matplotlib.pyplot as plt
from benchmark_functions import *
from optimization_algorithms import *


def main():
    # Tune parameters ----------------------------------------------------------
    dim_test = 2  # TODO: change dimensionality here
    # bounds = [(-5, 5), (-5, 5)] # Defines the search space for each dimension of the optimization problem.
    bounds = [(-5, 5)] * dim_test
    # n_agents = 5
    # n_agents = 10 # The number of agents or solutions in the population.
    n_agents = 50
    n_steps = 100  # The number of steps for the Lévy flight.
    beta = 1.5  # The stability parameter for the Lévy flight.
    scale = 0.01  # The scaling factor for the step size in the Lévy flight.
    max_gen = 50  # The maximum number of generations for the Differential Evolution algorithm.
    mut_factor = 0.5  # The mutation factor for the Differential Evolution algorithm.
    crossover_rate = 0.9  # The crossover rate for the Differential Evolution algorithm.
    # --------------------------------------------------------------------------

    f = rosenbrock

    # Create DE and FF optimizers for our eagle strategy
    DE = DifferentialEvolution(f, bounds, mut_factor, crossover_rate, n_agents=n_agents)
    FF = Firefly(f, bounds, beta0=0.5, gamma=1.0, alpha=1.0, theta=0.99, n_agents=n_agents)

    # Run the eagle strategy
    best_agent_DE, best_value_DE = eagle_strategy(
        f, DE, bounds, n_agents, n_steps, beta, scale, max_gen, mut_factor, crossover_rate
    )
    best_agent_FF, best_value_FF = eagle_strategy(
        f, FF, bounds, n_agents, n_steps, beta, scale, max_gen, mut_factor, crossover_rate
    )

    print(f"Best agent with DE: f({best_agent_DE})={best_value_DE:.10f}")
    print(f"Best agent with FF: f({best_agent_FF})={best_value_FF:.10f}")


if __name__ == "__main__":
    main()
