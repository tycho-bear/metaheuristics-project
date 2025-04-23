import numpy as np
from abc import ABC, abstractmethod
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
    sigma_u = (
        math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    path = np.zeros((n_steps, dim))
    pos = np.zeros(dim)

    for i in range(n_steps):
        u = np.random.normal(0, sigma_u, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / (np.abs(v) ** (1 / beta))
        pos += scale * step
        path[i] = pos

    return path


class OptimizationAlgorithm(ABC):
    """
    Base class for optimization algorithms
    """

    def set_agents(self, agents):
        """
        Sets the optimizer's agents to a predefined collection

        :param agents: (np.ndarray)
        """

        self.agents = agents

    def get_agents(self):
        """
        Get the optimizer's agents.

        :return: (np.ndarray)
        """

        return self.agents

    @abstractmethod
    def evolve(self, generations):
        pass


class DifferentialEvolution(OptimizationAlgorithm):
    def __init__(self, f, bounds, mut_factor=0.5, crossover_rate=0.9, n_agents=10):
        """
        Create a DE optimizer
        :param f: (function) Objective function to minimize.
        :param agents: (np.ndarray) Current population of agents.
        :param bounds: (list of tuples) Bounds for the solution space.
        :param mut_factor: (float) Mutation factor for DE.
        :param crossover_rate: (float) Crossover rate for DE.
        :param n_agents: (int) Number of agents in the population.
        """

        self.f = f
        self.bounds = np.array(bounds)
        self.mut_factor = mut_factor
        self.crossover_rate = crossover_rate
        self.n_agents = n_agents
        self.dim = len(bounds)

    def evolve(self, generations=1):
        """
        Perform DE for a given number of generations

        :param generations: (int) Number of generatiuons to evolve.
        """

        assert self.agents is not None

        f = self.f
        best = min(self.agents, key=f)

        for _ in range(generations):
            for i in range(self.n_agents):
                idxs = [idx for idx in range(self.n_agents) if idx != i]
                a, b, c = self.agents[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(
                    a + self.mut_factor * (b - c), [b[0] for b in self.bounds], [b[1] for b in self.bounds]
                )

                # binomial crossover
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.agents[i])

                if f(trial) < f(self.agents[i]):
                    self.agents[i] = trial

            best = min(best, *self.agents, key=f)

        return self.agents, best


class Firefly(OptimizationAlgorithm):
    def __init__(self, f, bounds, alpha=1.0, theta=0.95, beta0=0.5, gamma=0.01, n_agents=20):
        self.f = f
        self.bounds = np.array(bounds)
        self.alpha = alpha
        self.theta = theta
        self.beta0 = beta0
        self.gamma = gamma
        self.n_agents = n_agents
        self.dim = len(bounds)

    def _attractiveness(self, xi, xj):
        """
        Compute the attractiveness between two fireflies
        :param xi: First firefly
        :param xj: Second firefly
        :return: (float) Attractiveness
        """
        r = np.linalg.norm(xi - xj)
        return self.beta0 * np.exp(-self.gamma * r**2)

    def _move_firefly(self, i, j):
        """
        Move firefly i towards firefly j
        :param i: (int) index of first firefly
        :param j: (int) index of second firefly
        """
        beta = self._attractiveness(self.agents[i], self.agents[j])
        epsilon = np.random.uniform(-0.5, 0.5, self.dim)
        step = beta * (self.agents[j] - self.agents[i]) + self.alpha * epsilon
        self.agents[i] += step
        self.agents[i] = np.clip(self.agents[i], self.bounds[:, 0], self.bounds[:, 1])

    def evolve(self, generations=1):
        """
        Perform firefly optimization for a given number of generations

        :param generations: (int) Number of generatiuons to evolve.
        """

        assert type(self.agents) == type(np.array([]))

        f = self.f
        best = min(self.agents, key=f)
        alpha_p = self.alpha

        for _ in range(generations):
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if f(self.agents[i]) >= f(self.agents[j]):
                        self._move_firefly(i, j)

            alpha_p = alpha_p * self.theta

            best = min(best, *self.agents, key=f)

        return self.agents, best


def eagle_strategy(
    f, optimizer, global_bounds, n_agents=10, n_steps=100, beta=1.5, scale=0.01, max_gen=50, mut_factor=0.5, crossover_rate=0.9
):
    """
    Eagle strategy combining Levy flight for global search and Differential Evolution for local search.
    :param f: (function) Objective function to minimize.
    :param optimizer: (OptimizationAlgorithm) The local optimization algorithm to use
    :param global_bounds: (list of tuples) Bounds for the solution space.
    :param n_agents: (int) Number of agents in the population.
    :param n_steps: (int) Number of steps for Levy flight.
    :param beta: (float) Stability parameter for Levy flight.
    :param scale: (float) Step size scaling factor for Levy flight.
    :param max_gen: (int) Maximum number of generations for the algorithm.
    :param mut_factor: (float) Mutation factor for DE.
    :param crossover_rate: (float) Crossover rate for DE.
    :return: best_agent (np.ndarray), best_value (float)
    """

    dim = len(global_bounds)

    # Stage 1: Global search using Levy flight
    agents = np.random.uniform(*zip(*global_bounds), size=(n_agents, dim))
    best_agent = min(agents, key=f)

    for i in range(n_agents):
        path = levy_flight(n_steps, beta, scale, dim)
        move = path[-1]  # Last position in the flight path
        candidate = agents[i] + move
        candidate = np.clip(candidate, [b[0] for b in global_bounds], [b[1] for b in global_bounds])

        if f(candidate) < f(agents[i]):
            agents[i] = candidate
        if f(agents[i]) < f(best_agent):
            best_agent = agents[i]

    # Stage 2: Local search
    optimizer.set_agents(agents)
    _, best_agent = optimizer.evolve(max_gen)

    return best_agent, f(best_agent)
