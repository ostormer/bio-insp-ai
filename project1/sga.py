from mimetypes import init
import numpy as np
import pandas as pd
from math import sin, ceil, floor
from typing import Tuple
import random
from tqdm import tqdm
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
    assert n_selected % 2 == 0, "n_selected is odd. Need to select even number of parents."
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


def breed_parents(parents, mutation_chance) -> np.ndarray:
    shuffled = parents.copy()
    np.random.shuffle(shuffled)
    children = []
    for i in range(0, len(shuffled), 2):
        print(i)
        children.extend(crossover(shuffled[i], shuffled[i+1], mutation_chance))
    return np.array(children)


def survivor_selection(parents, offspring, pop_size, fitness_func, maximize=True) -> np.ndarray:
    parents_prob = find_fitness_prob(parents, fitness_func, maximize=maximize)
    offspring_prob = find_fitness_prob(
        offspring, fitness_func, maximize=maximize)
    # Implement elitism?
    # How many parents to keep?
    # Choose parents for breeding wih replacement?
    n_parents = floor(pop_size / 2)  # kill half, replace with offspring
    n_offspring = pop_size - n_parents
    parents_survivors = parents[np.random.choice(
        parents.shape[0], size=n_parents, p=parents_prob, replace=False)]
    offspring_survivors = offspring[np.random.choice(
        offspring.shape[0], size=n_offspring, p=offspring_prob, replace=False)]
    # TODO: Check axis of concat
    survivors = np.concatenate(parents_survivors, offspring_survivors, axis=0)
    return survivors


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


def sga(generations, pop_size, bitstring_length, fitness_func, mutation_chance):
    # initialize stuff
    initial_pop = generate_pop(pop_size, bitstring_length)
    pop = initial_pop.copy()
    # main loop
    n_parents = ceil(pop_size * 0.3) * 2
    for gen in range(generations):
        parents = parent_selection(pop, n_parents, fitness_func)
        offspring = breed_parents(parents, mutation_chance)
        pop = survivor_selection(parents, offspring, pop_size, fitness_func)
        # Evaluate and/or save result

    # end stuff
    
    # maybe return something


if __name__ == '__main__':
    data_df = pd.read_csv("data.csv")
    data = data_df.values
    n_features = data.shape[1]

    pop = generate_pop(40, 15)
    c1, c2 = crossover(pop[0], pop[1], 0.1)
    print("parents       children")
    print(pop[0], c1)
    print(pop[1], c2)
    print(sine(pop[0]))

    parents = parent_selection(pop, 20, sine)
    for ind in pop:
        print(ind, sine(ind))

    print()
    for ind in parents:
        print(ind, sine(ind))
    print()

    offspring = breed_parents(parents, 0.1)
    for ind in offspring:
        print(ind, sine(ind))
