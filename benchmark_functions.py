import numpy as np
import math


"""
Check Table 1 in the paper and see section 5 for this collection of functions
"""


#region Objective Functions
def ackley(x):
    """
    Python implementation of Ackley's function.

    The global minimum is located at the origin, i.e. x* = (0, ..., 0), with
    f(x*) = 0.

    x_i should be in [-35, 35].

    :param x: (np.ndarray) Input vector of dimension d.
    :return: (float) The function value at the given x.
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
    Python implementation of the Rosenbrock function.

    The global minimum is located at x* = f(1, ..., 1), f(x*) = 0.

    x_i should be in [-30, 30].

    :param x: (np.ndarray) Input vector of dimension 2.
    :return: (float) Function value.
    """


    # return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    D = len(x)

    summation = 0
    for i in range(D - 1):
        # xi = x[i]
        term = ((x[i] - 1) ** 2) + (100 * (x[i + 1] - x[i] ** 2) ** 2)
        summation += term

    return summation


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
    # sum_x = np.sum([i * np.cos(i + (i + 1) * x) for i in range(1, K + 1)])
    # return sum_x ** 2

    sum_x = 0
    sum_y = 0
    for i in range(K):
        sum_x += i * np.cos(i + (i + 1)*x[0])
        sum_y += i * np.cos(i + (i + 1)*x[1])

    return sum_x * sum_y




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


