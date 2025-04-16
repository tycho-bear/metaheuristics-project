import numpy as np
import math
import matplotlib.pyplot as plt
from benchmark_functions import *
from optimization_algorithms import eagle_strategy


def main():
    # Tune parameters ----------------------------------------------------------
    bounds = [(-5, 5), (-5, 5)] # Defines the search space for each dimension of the optimization problem.
    # n_agents = 10 # The number of agents or solutions in the population.
    n_agents = 50
    n_steps = 100 # The number of steps for the Lévy flight.
    beta = 1.5 # The stability parameter for the Lévy flight.
    scale = 0.01 # The scaling factor for the step size in the Lévy flight.
    max_gen = 50 # The maximum number of generations for the Differential Evolution algorithm.
    mut_factor = 0.5 # The mutation factor for the Differential Evolution algorithm.
    crossover_rate = 0.9 # The crossover rate for the Differential Evolution algorithm.
    # --------------------------------------------------------------------------


    # Run the eagle strategy
    best_agent, best_value = eagle_strategy(rosenbrock, bounds, n_agents, n_steps, beta, scale, max_gen, mut_factor,
                                            crossover_rate)

    print("Best agent:", best_agent)
    print(f"Best value: {best_value:.10f}")


if __name__ == "__main__":
    main()


