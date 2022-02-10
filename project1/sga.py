import numpy as np
import pandas as pd
from math import sin
from typing import Tuple
import random
# from .LinReg import LinReg


def generate_pop(n_individuals, bitstring_length) -> np.ndarray:
    pop = []
    for _ in range(n_individuals):
        pop.append(np.random.randint(2, size=bitstring_length))
    return np.array(pop)


def find_fitness_prob(pop, fitness_func, maximize=True) -> np.ndarray:
    """Finds probability of selecting each individual
     based on linearly transforming fitness

    Args:
        pop (np.ndarray): population of bitstrings
        fitness_func (function): fitness function to evaluate each bitstring
        maximize (bool, optional): Whether to maximize fitness.
            Minimizes if false. Defaults to True.

    Returns:
        np.ndarray: 1d array of probabilities
    """
    fitness = np.array([fitness_func(ind) for ind in pop])
    min_fitness = np.min(fitness)

    # Linearly rescale fitness to range [0, max-min]
    fitness_scaled = fitness - min_fitness
    s = np.sum(fitness_scaled)
    fitness_prob = fitness_scaled / s
    # Flip it if it should minimize fitness instead of maximizing
    if not maximize:
        fitness_prob = 1 - fitness_prob
    # TODO: Remove commented prints
    # for ind, fit, prob in zip(pop, fitness, fitness_prob):
    #     print(ind, fit, prob)
    return fitness_prob


def parent_selection(pop, n_selected, fitness_func, maximize=True) -> np.ndarray:
    """select parents from pop based on fitness_func

    Args:
        pop (np.ndarray): population, array of bitstrings
        n_selected (int): number of population to select as parents (0, 1)
        fitness_func (function): fitness function used to evaluate fitness
            of each individual. fitness_func(bitstring) -> fitness
        maximize (bool, optional): True if fitness function should be
            maximized. Minimizes if False. Defaults to True.
    """
    pop_prob = find_fitness_prob(pop, fitness_func, maximize=maximize)

    # select n_selected parents from pop
    parents = pop[np.random.choice(
        pop.shape[0], size=n_selected, p=pop_prob, replace=False)]

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
    """generates two offspring from two parents
    using uniform crossover and mutation

    Args:
        parent_a (np.ndarray): first parent
        parent_b (np.ndarray): second parent
        mutation_chance (float): chance of mutating,
            applied to children's every bit

    Returns:
        Tuple[np.ndarray, np.ndarray]: two children
    """
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


def survivor_selection(parents, offspring, pop_size) -> np.ndarray:
    parents_prob = find_fitness_prob(parents)
    offspring_prob = find_fitness_prob(offspring)
    # Implement elitism?
    # How many parents to keep?
    # Choose parents for breeding wih replacement?



def sine(x) -> float:
    """Converts bitstring x into number in range [0, 128]
        and returns sine of that number

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
    data_df = pd.read_csv("data.csv")
    data = data_df.values
    n_features = data.shape[1]

    pop = generate_pop(20, 6)
    c1, c2 = crossover(pop[0], pop[1], 0.1)
    print("parents       children")
    print(pop[0], c1)
    print(pop[1], c2)
    print(sine(pop[0]))

    parents = parent_selection(pop, 10, sine)
    for ind in pop:
        print(ind, sine(ind))

    print()
    for ind in parents:
        print(ind, sine(ind))
