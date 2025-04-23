import numpy as np
import math
import matplotlib.pyplot as plt
from benchmark_functions import *
from optimization_algorithms import *
import time



# Tune parameters ----------------------------------------------------------
seed = 4444
np.random.seed(seed)

# global parameters
f = rosenbrock
dim_test = 2  # TODO: change dimensionality here
# bounds = [(-5, 5), (-5, 5)] # Defines the search space for each dimension of the optimization problem.
global_bounds = [(-30, 30)] * dim_test
local_bounds = [(-5, 5)] * dim_test
# n_agents = 5
# n_agents = 10 # The number of agents or solutions in the population.
n_agents = 50

n_agents_values = [10, 25, 50, 100]


# Levy flight
n_steps = 100  # The number of steps for the Levy flight.
beta = 1.5  # The stability parameter for the Levy flight.
scale = 0.01  # The scaling factor for the step size in the Levy flight.

max_gen = 50  # The maximum number of generations for the Differential Evolution algorithm.
# differential evolution
mut_factor = 0.5  # The mutation factor for the Differential Evolution algorithm.
crossover_rate = 0.9  # The crossover rate for the Differential Evolution algorithm.

# firefly algorithm
beta0 = 0.5         # 0.5
gamma = 1           # 1
alpha = 1         # 1  (0.1 is better)
theta = 0.99        # 0.99

num_simulation_trials = 10  # do this number of trials and pick the best result
# --------------------------------------------------------------------------


def simulate_DE(DE, function):
    """

    Args:
        DE:
        function:

    Returns:

    """

    print("Running DE simulator...")
    print("Using these population sizes:", n_agents_values)
    print(f"Hyperparameters: ")

    start_time = time.time()

    for n_agents_num in n_agents_values:
        this_n_results = []
        for i in range(num_simulation_trials):
            # (best_agent, best_value) = eagle_strategy(f, DE, global_bounds,
            #                                           n_agents_num, n_steps,
            #                                           beta, scale, max_gen,
            #                                           mut_factor, crossover_rate)

            best_agent, best_value = eagle_strategy(
                f, DE, global_bounds, n_agents, n_steps, beta, scale, max_gen,
                mut_factor, crossover_rate
            )

            this_n_results.append((best_agent, best_value))
        # best todo
        # get the pair where best_value is the lowest in the list
        best_agent, best_value = min(this_n_results, key=lambda x: x[1])

        # print("Population size:", n_agents_num, "best found:\t\t", best_agent, " --> ", best_value)
        # same print but with a format string
        print(f"Population size:\t{n_agents_num}\t\tbest found:\t{best_agent} --> {best_value:.10f}")



    end_time = time.time()
    print(f"\nTime elapsed: {(end_time - start_time):.0f} seconds")






def main():




    # Create DE and FF optimizers for our eagle strategy
    DE = DifferentialEvolution(f, local_bounds, mut_factor, crossover_rate, n_agents=n_agents)
    FF = Firefly(f, local_bounds, beta0=beta0, gamma=gamma, alpha=alpha, theta=theta, n_agents=n_agents)


    simulate_DE(DE, f)



    # # Run the eagle strategy
    # best_agent_DE, best_value_DE = eagle_strategy(
    #     f, DE, global_bounds, n_agents, n_steps, beta, scale, max_gen, mut_factor, crossover_rate
    # )
    # print(f"Best agent with DE:  {best_agent_DE}  -->  {best_value_DE:.10f}")



    # best_agent_FF, best_value_FF = eagle_strategy(
    #     f, FF, global_bounds, n_agents, n_steps, beta, scale, max_gen, mut_factor, crossover_rate
    # )
    # print(f"Best agent with FF:  {best_agent_FF}  -->  {best_value_FF:.10f}")


if __name__ == "__main__":
    main()
