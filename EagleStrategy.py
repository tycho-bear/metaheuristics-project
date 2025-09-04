import numpy as np
import math
import matplotlib.pyplot as plt
from benchmark_functions import *
from optimization_algorithms import *
import time
import argparse
import sys



# Tune parameters ----------------------------------------------------------
seed = 666
np.random.seed(seed)

num_dimensions = 2  # TODO: change dimensionality here

DIFFERENTIAL_EVOLUTION_ALGORITHM = "de"
FIREFLY_ALGORITHM = "firefly"

ACKLEY_FUNCTION = "ackley"
SPHERE_FUNCTION = "sphere"
ROSENBROCK_FUNCTION = "rosenbrock"
SCHWEFEL_FUNCTION = "schwefel"
SHUBERT_FUNCTION = "shubert"

# The number of agents or solutions in the population.
n_agents = 50

n_agents_values = [5, 10, 25, 50, 100]

# Levy flight
n_steps = 100  # The number of steps for the Levy flight.
beta = 1.5  # The stability parameter for the Levy flight.
scale = 0.01  # The scaling factor for the step size in the Levy flight. # default was this

max_gen = 50  # 50 The maximum number of generations for the Differential Evolution algorithm.

# differential evolution
mut_factor = 0.5  # The mutation factor for the Differential Evolution algorithm. (F)
crossover_rate = 0.9  # The crossover rate for the Differential Evolution algorithm. (Cr)

global_bounds = None  # reassigned later

# firefly algorithm
# original
alpha = 0.1         # 1  (0.1 is better)
beta0 = 0.8         # 0.5
gamma = 1           # 1

theta = 0.99        # 0.99

num_simulation_trials = 10  # do this number of trials and pick the best result
# --------------------------------------------------------------------------


def main():
    # args: choose algorithm, choose function
    # function has corresponding global bounds
    # algorithm = de, firefly
    # function = ackley, sphere, rosenbrock, schwefel, shubert

    parser = argparse.ArgumentParser(
        description="Run optimization algorithm on a given problem."
    )
    parser.add_argument("algorithm", choices=["de", "firefly"],
                        help="The optimization algorithm to use: de or firefly.")
    parser.add_argument("function",
                        choices=["ackley", "sphere", "rosenbrock", "schwefel",
                                 "shubert"],
                        help="The function to minimize: ackley, sphere, "
                             "rosenbrock, schwefel, or shubert.")
    args = parser.parse_args()

    algorithm_arg = args.algorithm
    print("Algorithm argument:", algorithm_arg)

    global global_bounds  # reassign the variable outside this function

    # check function here and set global bounds
    function_arg = args.function
    if function_arg == ACKLEY_FUNCTION:
        function = ackley
        global_bounds = [(-32.768, 32.768)] * num_dimensions
    elif function_arg == SPHERE_FUNCTION:
        function = sphere
        global_bounds = [(-5.12, 5.12)] * num_dimensions
    elif function_arg == ROSENBROCK_FUNCTION:
        function = rosenbrock
        global_bounds = [(-5, 5)] * num_dimensions
    elif function_arg == SCHWEFEL_FUNCTION:
        function = schwefel
        global_bounds = [(-500, 500)] * num_dimensions
    else:
        function = shubert_single
        global_bounds = [(-10, 10)] * num_dimensions


    # run the algorithm + problem combination
    simulate(algorithm_arg, function)


def simulate(algorithm, function):
    """
    Runs the given optimization algorithm on the given function.

    :param algorithm: The optimization algorithm to use.
    :param function: The function to minimize.

    :return: None

    """

    # print("Running DE simulator...")
    # print("Using these population sizes:", n_agents_values)
    print()
    print(f"Algorithm: {algorithm}")
    print(f"Function: {function}")
    print(f"Global bounds: {global_bounds}")
    # print(f"Hyperparameters: F={mut_factor}, Cr={crossover_rate}")
    # print(f"Hyperparameters: alpha={alpha}, beta={beta0}, gamma={gamma}")
    print()

    start_time = time.time()

    for n_agents_num in n_agents_values:
        n_start_time = time.time()

        if algorithm == DIFFERENTIAL_EVOLUTION_ALGORITHM:
            opt = DifferentialEvolution(function, global_bounds, mut_factor,
                                        crossover_rate, n_agents=n_agents_num)
        else:
            opt = Firefly(function, global_bounds, beta0=beta0, gamma=gamma,
                          alpha=alpha, theta=theta, n_agents=n_agents_num)

        # opt = DifferentialEvolution(f, global_bounds, mut_factor, crossover_rate, n_agents=n_agents_num)
        # opt = Firefly(f, global_bounds, beta0=beta0, gamma=gamma, alpha=alpha, theta=theta, n_agents=n_agents_num)

        this_n_results = []

        for i in range(num_simulation_trials):
            best_agent, best_value = eagle_strategy(
                function, opt, global_bounds, n_agents_num, n_steps, beta, scale, max_gen,
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
        print(f"Population size:  {n_agents_num}    AVG best value found: --> {best_value:.8f}\t\t({duration:.3f} seconds)")

    end_time = time.time()
    print(f"Time elapsed: {(end_time - start_time):.1f} seconds")


if __name__ == "__main__":
    main()
