import numpy as np
import math
import matplotlib.pyplot as plt
import time
import itertools as it
import multiprocessing as mp
import random
from tqdm import tqdm
from functools import partial
from benchmark_functions import *
from optimization_algorithms import *


# Tune parameters ----------------------------------------------------------
seed = 666
np.random.seed(seed)

# global parameters

dim = 2  # TODO: change dimensionality here
# bounds = [(-5, 5), (-5, 5)] # Defines the search space for each dimension of the optimization problem.


# Ackley
ackley_global_bounds = [(-32.768, 32.768)] * dim
ackley_local_inradii = [5] * dim

# De Jong (sphere)
de_jong_global_bounds = [(-5.12, 5.12)] * dim
de_jong_local_inradii = [1] * dim

# Rosenbrock
rosenbrock_global_bounds = [(-5, 5)] * dim
rosenbrock_local_inradii = [1] * dim

# Schwefel
schwefel_global_bounds = [(-500, 500)] * dim
schwefel_local_inradii = [100] * dim

# Shubert single
shubert_global_bounds = [(-10, 10)] * dim
shubert_local_inradii = [1.5] * dim

fs_bounds = {
    "ackley": (ackley, ackley_global_bounds, ackley_local_inradii),
    "de_jong": (sphere, de_jong_global_bounds, de_jong_local_inradii),
    "rosenbrock": (rosenbrock, rosenbrock_global_bounds, rosenbrock_local_inradii),
    "schwefel": (schwefel, schwefel_global_bounds, schwefel_local_inradii),
    "shubert_single": (shubert_single, shubert_global_bounds, shubert_local_inradii),
}

# Function determiner
F = "shubert_single"
f, global_bounds, local_inradii = fs_bounds[F]


# n_agents = 5
# n_agents = 10 # The number of agents or solutions in the population.
n_agents = 50

n_agents_values = [5, 10, 25, 50, 100]


# Levy flight
n_steps = 100  # The number of steps for the Levy flight.
beta = 1.5  # The stability parameter for the Levy flight.
scale = 0.01  # The scaling factor for the step size in the Levy flight.

max_gen = 50  # 50 The maximum number of generations for the Differential Evolution algorithm.
# differential evolution
# mut_factor = 0.5  # The mutation factor for the Differential Evolution algorithm. (F)
# crossover_rate = 0.9  # The crohssover rate for the Differential Evolution algorithm. (Cr)
mut_factor = 0.8
crossover_rate = 0.7


# firefly algorithm
alpha = 0.1 # Randomization parameter for FA
theta = 0.15 # Randomization decay for FA
beta0 = 1. # Attractivenehss at r=0 for FA
gamma = 0.9 # Light Absorption Coefficient for FA

hyperparam_search_size = 0 # 100000

num_simulation_trials = 10  # do this number of trials and pick the best result
# --------------------------------------------------------------------------


def simulate_DE(function):
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
    print(f"Local bounds: {local_inradii}")
    print(f"Hyperparameters: F={mut_factor}, Cr={crossover_rate}")
    print()

    start_time = time.time()

    for n_agents_num in n_agents_values:
        n_start_time = time.time()
        DE = DifferentialEvolution(
            f, dim, mut_factor, crossover_rate, n_agents=n_agents_num
        )
        this_n_results = []
        for i in range(num_simulation_trials):
            # (best_agent, best_value) = eagle_strategy(f, DE, global_bounds, local_inradii
            #                                           n_agents_num, n_steps,
            #                                           beta, scale, max_gen,)

            best_agent, best_value = eagle_strategy(
                f,
                DE,
                global_bounds,
                local_inradii,
                n_agents_num,
                n_steps,
                beta,
                scale,
                max_gen,
            )

            this_n_results.append((best_agent, best_value))

        # get the average best_value found to avoid outliers
        # best_value = np.mean([x[1] for x in this_n_results])

        # find the best
        best_agent, best_value = min(this_n_results, key=lambda x: x[1])
        n_end_time = time.time()
        duration = n_end_time - n_start_time

        print(
            f"Population size:\t{n_agents_num}\t\tbest found:\t{best_agent} --> {best_value:.8f}\t\t({duration:.3f} seconds)"
        )
        # print(f"Population size:\t{n_agents_num}\t\tAVG best value found: --> {best_value:.8f}\t\t({duration:.3f} seconds)")

    end_time = time.time()
    print(f"Time elapsed: {(end_time - start_time):.1f} seconds")


def train_firefly_meta(function, n_agents, max_gen, i):
    alpha = random.uniform(0, 1)
    theta = random.uniform(0, 1)
    beta0 = random.uniform(0, 1)
    gamma = random.uniform(0, 1)
    fa = Firefly(function, dim, alpha, theta, beta0, gamma, n_agents)
    _, score = eagle_strategy(f, fa, global_bounds, local_inradii, n_agents, n_steps, beta, scale, max_gen)
    return (alpha, theta, beta0, gamma, n_agents, max_gen), score


def simulate_FA(function):
    print()
    print("Running FA simulator...")
    print(f"Function: {function}")
    print(f"Global bounds: {global_bounds}")
    print(f"Local bounds: {local_inradii}")

    al, th, b0, ga = alpha, theta, beta0, gamma
    hss = hyperparam_search_size
    if hss:
        print("Performing random metavariable search...")
        # Use lower n_agents and max_gen for search
        train_meta = partial(train_firefly_meta, function, 25, num_simulation_trials)
        cpus = mp.cpu_count()
        chunksize = min(hss // (cpus * 10), 256)
        with mp.Pool(cpus) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(train_meta, range(hss), chunksize=chunksize),
                    total=hss
                )
            )

        params, score = min(results, key=lambda x: x[1])
        al, th, b0, ga, _, _ = params

        print(f"Found the following best hyperparams (with score {score}):")

    else:
        print("Hyperparameters:")

    print(f"alpha: {al}")
    print(f"theta: {th}")
    print(f"beta0: {b0}")
    print(f"gamma: {ga}")

    start_time = time.time()

    for n_agents_num in n_agents_values:
        n_start_time = time.time()
        FA = Firefly(f, dim, al, th, b0, ga, n_agents_num)
        this_n_results = []
        for i in range(num_simulation_trials):
            best_agent, best_value = eagle_strategy(
                f, FA, global_bounds, local_inradii, n_agents_num, n_steps, beta, scale, max_gen
            )

            this_n_results.append((best_agent, best_value))

        # get the average best_value found to avoid outliers
        # best_value = np.mean([x[1] for x in this_n_results])

        # find the best
        best_agent, best_value = min(this_n_results, key=lambda x: x[1])
        n_end_time = time.time()
        duration = n_end_time - n_start_time

        print(
            f"Population size:\t{n_agents_num}\t\tbest found:\t{best_agent} --> {best_value:.8f}\t\t({duration:.3f} seconds)"
        )
    end_time = time.time()
    print(f"Time elapsed: {(end_time - start_time):.1f} seconds")


def main():
    # Create DE and FF optimizers for our eagle strategy
    # DE = DifferentialEvolution(f, dim, mut_factor, crossover_rate, n_agents=n_agents)
    # FF = Firefly(
    #     f, dim, beta0=beta0, gamma=gamma, alpha=alpha, theta=theta, n_agents=n_agents
    # )

    simulate_DE(f)
    simulate_FA(f)

    # # Run the eagle strategy
    # best_agent_DE, best_value_DE = eagle_strategy(
    #     f, DE, global_bounds, local_inradii, n_agents, n_steps, beta, scale, max_gen
    # )
    # print(f"Best agent with DE:  {best_agent_DE}  -->  {best_value_DE:.10f}")

    # best_agent_FF, best_value_FF = eagle_strategy(
    #     f, FF, global_bounds, local_inradii, n_agents, n_steps, beta, scale, max_gen
    # )
    # print(f"Best agent with FF:  {best_agent_FF}  -->  {best_value_FF:.10f}")


if __name__ == "__main__":
    main()
