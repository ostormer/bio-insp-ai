import numpy as np
import pandas as pd
from math import sin
from typing import Tuple
import random
# from .LinReg import LinReg

data_df = pd.read_csv("data.csv")
data = data_df.values
print(data.shape)
n_features = data.shape[1]


def generate_pop(n_individuals, bitstring_length) -> np.ndarray:
    pop = []
    for _ in range(n_individuals):
        pop.append(np.random.randint(2, size=bitstring_length))
    return np.array(pop)


def parent_selection(pop, n_selected, fitness_func, maximize=True):
    """select parents from pop based on fitness_func

    Args:
        pop (np.ndarray): population, array of bitstrings
        n_selected (int): number of population to select as parents (0, 1)
        fitness_func (function): fitness function used to evaluate fitness of each individual. fitness_func(bitstring) -> fitness
        maximize (bool, optional): True if fitness function should be maximized. Minimizes if False. Defaults to True.
    """
    fitness = []
    min_fitness = float('inf')
    max_fitness = float('-inf')
    for individual in pop:
        # Save each indivividual's fitness
        fitness.append(fitness_func(individual))
        # Update min and max values for later scaling
        if fitness[-1] < min_fitness:
            min_fitness = fitness[-1]
        if fitness[-1] > max_fitness:
            max_fitness = fitness[-1]

    # Linearly rescale fitness to range [0, max-min]
    s = 0
    for i in range(len(pop)):
        fitness[i] = fitness[i] - min_fitness
        s += fitness[i]

    pop_prob = []
    # Rescale fitness to probabilities
    for i in range(len(pop)):
        pop_prob.append(fitness[i] / s)

    print(sum(pop_prob))

    # select n_selected parents from pop
    parents = np.random.choice(pop, size=n_selected, p=pop_prob)

    return parents


def flip_bit(n) -> int:
    """turn 1 to 0 or 0 to 1

    Args:
        n (int): bit (0 or 1) to be flipped

    Returns:
        int: flipped bit
    """
    assert n == 0 or n == 1, "n is not 0 or 1"
    return 1 if n == 0 else 0


def crossover(parent_a, parent_b, mutation_chance) -> Tuple[np.ndarray, np.ndarray]:
    child_a = np.zeros_like(parent_a)
    child_b = np.zeros_like(parent_a)
    # Uniform crossover, 50/50 chance
    exchange_bits = np.random.randint(2, size=parent_a.shape)
    for i, (a, b, exchange) in enumerate(zip(parent_a, parent_b, exchange_bits)):
        if exchange:
            child_a[i] = b
            child_b[i] = a
        else:
            child_a[i] = a
            child_b[i] = b

        # Mutate child a
        if random.random() < mutation_chance:
            child_a[i] = flip_bit(child_a[i])
        # Mutate child b
        if random.random() < mutation_chance:
            child_b[i] = flip_bit(child_b[i])

    return child_a, child_b


def sine(x) -> float:
    """Converts bitstring x into number in range [0, 128] and returns sine of that number

    Args:
        x (numpy array): bitstring

    Returns:
        float: sine of bitstring
    """
    s = 0
    for i, bit in enumerate(x):
        s += bit * (2 ** (len(x) - 1 - i))

    scaling_factor = 2 ** (7 - len(x))
    return sin(s * scaling_factor)


if __name__ == '__main__':
    pop = generate_pop(20, 4)
    print(sine(np.array([1, 0, 1, 1, 1, 1, 1])))
