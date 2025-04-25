import numpy as np
import math
import matplotlib.pyplot as plt
from benchmark_functions import *
from optimization_algorithms import *
import time



# Tune parameters ----------------------------------------------------------
seed = 666
np.random.seed(seed)

dim_test = 2  # TODO: change dimensionality here
# bounds = [(-5, 5), (-5, 5)] # Defines the search space for each dimension of the optimization problem.

# f = ackley
# f = sphere
# f = rosenbrock
# f = schwefel
f = shubert_single


# global_bounds = [(-32.768, 32.768)] * dim_test  # Ackley
# global_bounds = [(-5.12, 5.12)] * dim_test      # De Jong (sphere)
# global_bounds = [(-5, 5)] * dim_test            # Rosenbrock
# global_bounds = [(-500, 500)] * dim_test          # Schwefel
global_bounds = [(-10, 10)] * dim_test          # Shubert single

# Ackley
# global_bounds = [(-32.768, 32.768)] * dim_test
# local_bounds = [(-5, 5)] * dim_test

# De Jong (sphere)
# global_bounds = [(-5.12, 5.12)] * dim_test
# local_bounds = [(-1, 1)] * dim_test

# Rosenbrock
# global_bounds = [(-5, 5)] * dim_test
# local_bounds = [(-1, 1)] * dim_test

# Schwefel
# global_bounds = [(-500, 500)] * dim_test
# # local_bounds = [(-75, 75)] * dim_test
# local_bounds = global_bounds

# Shubert single
# global_bounds = [(-10, 10)] * dim_test
# local_bounds = [(-1.5, 1.5)] * dim_test



# n_agents = 5
# n_agents = 10 # The number of agents or solutions in the population.
n_agents = 50

n_agents_values = [5, 10, 25, 50, 100]


# Levy flight
n_steps = 100  # The number of steps for the Levy flight.
beta = 1.5  # The stability parameter for the Levy flight.
# beta = 1  # The stability parameter for the Levy flight.
scale = 0.01  # The scaling factor for the step size in the Levy flight. # default was this

max_gen = 50  # 50 The maximum number of generations for the Differential Evolution algorithm.
# differential evolution
mut_factor = 0.5  # The mutation factor for the Differential Evolution algorithm. (F)
crossover_rate = 0.9  # The crossover rate for the Differential Evolution algorithm. (Cr)
# mut_factor = 0.8
# crossover_rate = 0.7


# firefly algorithm
# original
alpha = 0.1         # 1  (0.1 is better)
beta0 = 0.8         # 0.5
gamma = 1           # 1
# alpha = 0.2         # 1  (0.1 is better)
# beta0 = 0.9         # 0.5
# gamma = 2           # 1

# bad
# alpha = 0.6
# beta0 = 0.4
# gamma = 1



theta = 0.99        # 0.99
# theta = 0.9        # trying this

num_simulation_trials = 10  # do this number of trials and pick the best result
# --------------------------------------------------------------------------


def simulate(function):
    """

    Args:
        DE:
        function:

    Returns:

    """

    # print("Running DE simulator...")
    # print("Using these population sizes:", n_agents_values)
    print()
    print(f"Function: {function}")
    print(f"Global bounds: {global_bounds}")
    # print(f"Local bounds: {local_bounds}")
    # print(f"Hyperparameters: F={mut_factor}, Cr={crossover_rate}")
    print(f"Hyperparameters: alpha={alpha}, beta={beta0}, gamma={gamma}")
    print()

    start_time = time.time()

    for n_agents_num in n_agents_values:
        n_start_time = time.time()

        # opt = DifferentialEvolution(f, global_bounds, mut_factor, crossover_rate, n_agents=n_agents_num)
        # opt = Firefly(f, global_bounds, mut_factor, crossover_rate, n_agents=n_agents_num)
        opt = Firefly(f, global_bounds, beta0=beta0, gamma=gamma, alpha=alpha, theta=theta, n_agents=n_agents_num)

        this_n_results = []
        for i in range(num_simulation_trials):
            # (best_agent, best_value) = eagle_strategy(f, DE, global_bounds,
            #                                           n_agents_num, n_steps,
            #                                           beta, scale, max_gen,
            #                                           mut_factor, crossover_rate)

            best_agent, best_value = eagle_strategy(
                f, opt, global_bounds, n_agents_num, n_steps, beta, scale, max_gen,
                mut_factor, crossover_rate
            )



            this_n_results.append((best_agent, best_value))

        # get the average best_value found to avoid outliers
        best_value = np.mean([x[1] for x in this_n_results])


        # find the best
        # best_agent, best_value = min(this_n_results, key=lambda x: x[1])
        n_end_time = time.time()
        duration = n_end_time - n_start_time

        # print(f"Population size:\t{n_agents_num}\t\tbest found:\t{best_agent} --> {best_value:.8f}\t\t({duration:.3f} seconds)")
        print(f"Population size:\t{n_agents_num}\t\tAVG best value found: --> {best_value:.8f}\t\t({duration:.3f} seconds)")



    end_time = time.time()
    print(f"Time elapsed: {(end_time - start_time):.1f} seconds")






def main():




    # Create DE and FF optimizers for our eagle strategy
    # DE = DifferentialEvolution(f, local_bounds, mut_factor, crossover_rate, n_agents=n_agents)
    # FF = Firefly(f, local_bounds, beta0=beta0, gamma=gamma, alpha=alpha, theta=theta, n_agents=n_agents)


    simulate(f)



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
