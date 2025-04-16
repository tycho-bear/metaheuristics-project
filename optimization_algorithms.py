import numpy as np
import math

def levy_flight(n_steps=1000, beta=1.5, scale=0.01, dim=2):
    """
    Generate a Levy flight using Mantegna's algorithm.

    :param n_steps: (int) Number of steps in the flight.
    :param beta: (float) Stability parameter (0 < beta ≤ 2).
    :param scale: (float) Step size scaling factor.
    :param dim: (int) Number of dimensions to walk in (default is 2, so 2D).
    :return: (np.ndarray) Array of positions in the flight path. Format is
        (n_steps, dim).
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

        # binomial crossover
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
