import numpy as np
import random
from typing import Tuple
from tqdm import tqdm
from math import floor, ceil
from copy import deepcopy


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
    # Epsilon > 0 so that every individual has a small chance of being chosen
    eps = 1 / len(pop) * 1e-6
    # Linearly rescale fitness to range [eps, max-min+eps]
    fitness_scaled = fitness - min_fitness + eps
    # Flip it if it should minimize fitness instead of maximizing
    if not maximize:
        fitness_scaled = -fitness_scaled
    s = np.sum(fitness_scaled)
    fitness_prob = fitness_scaled / s
    return fitness_prob


def parent_selection(
    pop,
    n_selected,
    fitness_func,
    maximize=True
) -> np.ndarray:
    """select parents from pop based on fitness_func

    Args:
        pop (np.ndarray): population, array of bitstrings
        n_selected (int): number of population to select as parents (0, 1)
        fitness_func (function): fitness function used to evaluate fitness
            of each individual. fitness_func(bitstring) -> fitness
        maximize (bool, optional): True if fitness function should be
            maximized. Minimizes if False. Defaults to True.
    """
    assert n_selected % 2 == 0, \
        "n_selected is odd. Need to select even number of parents."
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


def uniform_crossover(
    parent_a,
    parent_b,
    mutation_chance
) -> Tuple[np.ndarray, np.ndarray]:
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
        children.extend(uniform_crossover(
            shuffled[i], shuffled[i+1], mutation_chance))
    return np.array(children)


def survivor_selection(
    parents, offspring,
    pop_size,
    fitness_func,
    maximize=True
) -> np.ndarray:
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

    survivors = np.concatenate(
        (parents_survivors, offspring_survivors), axis=0)
    return survivors


def bitstring_to_num(x) -> np.ndarray:
    """Converts bitstring x into number in range [0, 128]
    Args:
        x (numpy array): bitstring

    Returns:
        float: real value of bitstring in range [0, 128]
    """
    s = 0
    for i, bit in enumerate(x):
        s += bit * (2 ** (len(x) - 1 - i))

    scaling_factor = 2 ** (7 - len(x))
    return s * scaling_factor


def sga(
    generations,
    pop_size,
    bitstring_length,
    fitness_func,
    mutation_chance,
    maximize_fitness=True,
) -> None:
    # initialize stuff
    pop = generate_pop(pop_size, bitstring_length)
    pop_history = []
    pop_history.append(deepcopy(pop))
    fitness_history = []
    fitness_history.append(np.array([fitness_func(ind) for ind in pop]))
    # main loop
    n_parents = ceil(pop_size * 0.3) * 2
    print("Running Simple Genetic Algorithm for {:d} generations:".format(
        generations))
    for _ in tqdm(range(generations)):
        parents = parent_selection(
            pop, n_parents, fitness_func, maximize=maximize_fitness)
        offspring = breed_parents(parents, mutation_chance)
        pop = survivor_selection(
            parents,
            offspring,
            pop_size,
            fitness_func,
            maximize=maximize_fitness
        )
        fitness_history.append(np.array([fitness_func(ind) for ind in pop]))
        pop_history.append(deepcopy(pop))  # Save a copy of pop

    # end stuff

    # maybe return something
    return pop_history, fitness_history
