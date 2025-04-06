import numpy as np
import math
import matplotlib.pyplot as plt

#region Objective Functions
def ackley(x):
    """
    Ackley function for optimization global optimum f = 0 x = (0,0,....0).
    :param x: (np.ndarray) Input vector of dimension d.
    :return: (float) Function value.
    """
    d = len(x) # Number of dimensions
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / d) + 20 + np.e
    return term1 + term2

def sphere(x):
    """
    Sphere (simple De Jong) function for optimization global optimum f = 0 x = (0,0,....0).
    :param x: (np.ndarray) Input vector of dimension d.
    :return: (float) Function value.
    """
    return np.sum(np.square(x))

def rosenbrock(x):
    """
    Rosenbrock function for optimization global optimum f = 0 x = (1,1)..
    :param x: (np.ndarray) Input vector of dimension 2.
    :return: (float) Function value.
    """
    return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def schwefel(x):
    """
    Schwefel function for optimization global optimum f = 0 x = (420.9687,420.9687,....420.9687).
    :param x: (np.ndarray) Input vector of dimension d.
    :return: (float) Function value.
    """
    d = len(x)
    return 418.9829 * d - np.sum([x[i] * np.sin(np.sqrt(abs(x[i]))) for i in range(d)])

def shubert_single(x, K=5):
    """
    Shubert function for optimization with a single input global optimum f = -186.7309 x = (-1,1).
    :param x: (np.ndarray) Input vector.
    :param K: (int) Number of terms in the Shubert function.
    :return: (float) Function value.
    """
    sum_x = np.sum([i * np.cos(i + (i + 1) * x) for i in range(1, K + 1)])
    return sum_x ** 2

def shubert_multi(x, y, K=5):
    # TODO: Will need to modify other code to accmadate this function if we want to use it
    """
    Shubert function for optimization with two inputs global optimum f = -186.7309 x = (-1,1).
    :param x: (float) First input.
    :param y: (float) Second input.
    :param K: (int) Number of terms in the Shubert function.
    :return: (float) Function value.
    """
    sum_x = np.sum([i * np.cos(i + (i + 1) * x) for i in range(1, K + 1)])
    sum_y = np.sum([i * np.cos(i + (i + 1) * y) for i in range(1, K + 1)])
    return sum_x * sum_y




#endregion

def levy_flight(n_steps=1000, beta=1.5, scale=0.01, dim=2):
    """
    Generate a Lévy flight using Mantegna's algorithm.

    Parameters:
        n_steps (int): Number of steps in the flight.
        beta (float): Stability parameter (0 < beta ≤ 2).
        scale (float): Step size scaling factor.
        dim (int): Dimensionality of the walk (default is 2D).

    Returns:
        np.ndarray: (n_steps, dim) array of positions in the flight path.
    """
    # Mantegna's algorithm
    sigma_u = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)

    path = np.zeros((n_steps, dim))
    pos = np.zeros(dim)

    for i in range(n_steps):
        u = np.random.normal(0, sigma_u, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / (np.abs(v) ** (1 / beta))
        pos += scale * step
        path[i] = pos

    return path

def differential_evolution(f, agents, bounds, mut_factor=0.5, crossover_rate=0.9, n_agents=10):
    """
    Perform Differential Evolution (DE) for local search.
    :param f: (function) Objective function to minimize.
    :param agents: (np.ndarray) Current population of agents.
    :param bounds: (list of tuples) Bounds for the solution space.
    :param mut_factor: (float) Mutation factor for DE.
    :param crossover_rate: (float) Crossover rate for DE.
    :param n_agents: (int) Number of agents in the population.
    :return: (np.ndarray) Updated population of agents after DE step.
    """

    dim = agents.shape[1]

    for i in range(n_agents):
        idxs = [idx for idx in range(n_agents) if idx != i]
        a, b, c = agents[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + mut_factor * (b - c), [b[0] for b in bounds], [b[1] for b in bounds])

        cross_points = np.random.rand(dim) < crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, dim)] = True

        trial = np.where(cross_points, mutant, agents[i])

        if f(trial) < f(agents[i]):
            agents[i] = trial

    return agents


def eagle_strategy(f, bounds, n_agents=10, n_steps=100, beta=1.5, scale=0.01, max_gen=50, mut_factor=0.5,
                   crossover_rate=0.9):
    """
    Eagle strategy combining Levy flight for global search and Differential Evolution for local search.
    :param f: (function) Objective function to minimize.
    :param bounds: (list of tuples) Bounds for the solution space.
    :param n_agents: (int) Number of agents in the population.
    :param n_steps: (int) Number of steps for Levy flight.
    :param beta: (float) Stability parameter for Levy flight.
    :param scale: (float) Step size scaling factor for Levy flight.
    :param max_gen: (int) Maximum number of generations for the algorithm.
    :param mut_factor: (float) Mutation factor for DE.
    :param crossover_rate: (float) Crossover rate for DE.
    :return: best_agent (np.ndarray), best_value (float)
    """

    dim = len(bounds)

    # Stage 1: Global search using Levy flight
    agents = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(n_agents, dim))
    best_agent = min(agents, key=f)

    for i in range(n_agents):
        path = levy_flight(n_steps, beta, scale, dim)
        move = path[-1]  # Last position in the flight path
        candidate = agents[i] + move
        candidate = np.clip(candidate, [b[0] for b in bounds], [b[1] for b in bounds])

        if f(candidate) < f(agents[i]):
            agents[i] = candidate
        if (f(agents[i]) < f(best_agent)):
            best_agent = agents[i]


    # Stage 2: Local search using Differential Evolution
    for gen in range(max_gen):
        agents = differential_evolution(f, agents, bounds, mut_factor, crossover_rate, n_agents)

        # Update best agent
        for agent in agents:
            if f(agent) < f(best_agent):
                best_agent = agent

    return best_agent, f(best_agent)


def main():
    # Tune parameters ----------------------------------------------------------
    bounds = [(-5, 5), (-5, 5)] # Defines the search space for each dimension of the optimization problem.
    n_agents = 10 # The number of agents or solutions in the population.
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
    print("Best value:", best_value)


if __name__ == "__main__":
    main()


