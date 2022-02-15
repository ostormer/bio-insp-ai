import numpy as np
import pandas as pd
import seaborn as sns
import random
import os
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm import tqdm
from math import floor, ceil, log2
from copy import deepcopy


def generate_pop(n_individuals, bitstring_length) -> np.ndarray:
    pop = []
    for _ in range(n_individuals):
        pop.append(np.random.randint(2, size=bitstring_length))
    return np.array(pop)


def find_fitness_prob(pop, fitness_func, maximize_fitness=True) -> np.ndarray:
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
    max_fitness = np.max(fitness)
    # Epsilon > 0 so that every individual has a small chance of being chosen
    # eps = 1 / len(pop) * 1e-6
    # Linearly rescale fitness to range [eps, max-min+eps]
    if maximize_fitness:
        fitness_scaled = fitness - min_fitness  # + eps
    else:
        # Flip it if it should minimize fitness instead of maximizing
        fitness_scaled = max_fitness - fitness  # + eps
    fitness_scaled = fitness_scaled / np.max(fitness_scaled)
    # This exp scaling makes eps unneeded, as 0 is turned into -1**50 > 0
    fitness_scaled = 50 ** (fitness_scaled - 1)
    s = np.sum(fitness_scaled)
    fitness_prob = fitness_scaled / s
    return fitness_prob


def parent_selection(
    pop,
    n_selected,
    fitness_func,
    maximize_fitness=True
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
    pop_prob = find_fitness_prob(
        pop, fitness_func, maximize_fitness=maximize_fitness)

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
    for i, (a, b, exchange) in enumerate(
        zip(parent_a, parent_b, exchange_bits)
    ):
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


def breed_all_parents(parents, mutation_chance) -> np.ndarray:
    shuffled = parents.copy()
    np.random.shuffle(shuffled)
    children = []
    for i in range(0, len(shuffled), 2):
        children.extend(uniform_crossover(
            shuffled[i], shuffled[i+1], mutation_chance))
    return np.array(children)


def breed_parents_with_replacement(
    pop,
    fitness_func,
    n_offspring,
    mutation_chance,
    maximize_fitness=True
) -> np.ndarray:
    assert n_offspring % 2 == 0
    parents_prob = find_fitness_prob(
        pop, fitness_func, maximize_fitness=maximize_fitness)
    offspring = []
    for _ in range(n_offspring // 2):
        parents = pop[np.random.choice(
            pop.shape[0], size=2, p=parents_prob, replace=True)]
        offspring.extend(uniform_crossover(
            parents[0], parents[1], mutation_chance))
    return np.array(offspring)


def survivor_selection(
    parents, offspring,
    pop_size,
    fitness_func,
    maximize_fitness=True
) -> np.ndarray:
    """select survivors from parents and offspring

    Args:
        parents (np.ndarray): 2d array of parent bitstrings
        offspring (np.ndarray): 2d array of offspring bitstrings
        pop_size (int): population size
        fitness_func (function): fitness function
        maximize_fitness (bool, optional): Whether to maximize fitness value.
            Defaults to True.

    Returns:
        np.ndarray: Array of survivors
    """
    # Implement elitism?

    parents_prob = find_fitness_prob(
        parents, fitness_func, maximize_fitness=maximize_fitness)
    offspring_prob = find_fitness_prob(
        offspring, fitness_func, maximize_fitness=maximize_fitness)
    # How many parents to keep?
    # Choose parents for breeding wih replacement?
    n_parents = floor(pop_size * 0.4)  # kill 0.6, replace with offspring
    n_offspring = pop_size - n_parents
    assert n_offspring <= len(offspring), "Not enough offspring generated"
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
    breed_with_replacement=False,
    crowding=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple Genetic Algorithm

    Args:
        generations (int): number of generations
        pop_size (int): population size, must be even
        bitstring_length (int): length of bitstring per individual
        fitness_func (function): Fitness function
        mutation_chance (float): mutation chance in interval (0, 1)
        maximize_fitness (bool, optional): Whether to maximize the fitness
            function. Defaults to True.
        breed_with_replacement (bool, optional): Whether to let single parent
            breed multiple times per generation. Defaults to False.
        crowding (bool, optional): Whether to use deterministic crowding
            instead of standard SGA. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: pop_hist, fitness_hist
    """
    # initialize stuff
    assert pop_size % 2 == 0
    pop = generate_pop(pop_size, bitstring_length)
    pop_history = []
    pop_history.append(deepcopy(pop))
    fitness_history = []
    fitness_history.append(np.array([fitness_func(ind) for ind in pop]))

    # number of parents to select when each parent can breed once
    n_parents = ceil(pop_size * 0.3) * 2
    print("Running Simple Genetic Algorithm for {:d} generations:".format(
        generations))
    for _ in tqdm(range(generations)):
        parents = pop
        if not crowding:
            if not breed_with_replacement:
                parents = parent_selection(pop, n_parents, fitness_func,
                                           maximize_fitness=maximize_fitness)
                offspring = breed_all_parents(parents, mutation_chance)
            else:
                offspring = breed_parents_with_replacement(
                    pop,
                    fitness_func,
                    n_offspring=pop_size,
                    mutation_chance=mutation_chance,
                    maximize_fitness=maximize_fitness
                )
            pop = survivor_selection(
                parents,
                offspring,
                pop_size,
                fitness_func,
                maximize_fitness=maximize_fitness
            )
            fitness_history.append(
                np.array([fitness_func(ind) for ind in pop]))
        else:
            # Deterministic crowding
            # Two parents selected at random without replacement,
            # Fitness evaluation only in child vs parent competition
            np.random.shuffle(parents)
            survivors = []
            survivors_fitness = []
            max_fit = 1 if maximize_fitness else -1
            for i in range(pop_size // 2):
                p1 = parents[i]
                p2 = parents[i + 1]
                c1, c2 = uniform_crossover(p1, p2, mutation_chance)
                # Compute hamming distances of both possible p+c pairings
                dist_same = np.count_nonzero(
                    p1 != c1) + np.count_nonzero(p2 != c2)
                dist_cross = np.count_nonzero(
                    p1 != c2) + np.count_nonzero(p2 != c1)
                # Parent-child tournament. Each parent competes against its
                # "closest" child. (Both parents and both children competes
                # once, the pairing with smallest total distance is chosen)
                fp1 = fitness_func(p1)
                fp2 = fitness_func(p2)
                fc1 = fitness_func(c1)
                fc2 = fitness_func(c2)
                if dist_same < dist_cross:
                    # p1 vs c1
                    if max_fit * fp1 > max_fit * fc1:
                        survivors.append(p1)
                        survivors_fitness.append(fp1)
                    else:
                        survivors.append(c1)
                        survivors_fitness.append(fc1)
                    # p2 vs c2
                    if max_fit * fp2 > max_fit * fc2:
                        survivors.append(p2)
                        survivors_fitness.append(fp2)
                    else:
                        survivors.append(c2)
                        survivors_fitness.append(fc2)
                else:
                    # p1 vs c2
                    if max_fit * fp1 > max_fit * fc2:
                        survivors.append(p1)
                        survivors_fitness.append(fp1)
                    else:
                        survivors.append(c2)
                        survivors_fitness.append(fc2)
                    # p2 vs c1
                    if max_fit * fp2 > max_fit * fc1:
                        survivors.append(p2)
                        survivors_fitness.append(fp2)
                    else:
                        survivors.append(c1)
                        survivors_fitness.append(fc1)
            fitness_history.append(np.array(survivors_fitness))
            pop = np.array(survivors)

        pop_history.append(deepcopy(pop))  # Save a copy of pop

    # end stuff
    return pop_history, fitness_history


def fitness_box_plot(fitness_hist):
    """plot and show a box plot of each generation's fitness

    Args:
        fitness_hist (list[np.ndarray]): List of nd.array of each
            individual's fitness for each generation
    """
    sns.set_theme(style="whitegrid")
    fitness_df = pd.DataFrame()
    for gen, pop_fitness in enumerate(fitness_hist):
        fitness_df[gen] = pop_fitness

    sns.boxplot(data=fitness_df)
    plt.show()


def entropy_plot(pop_histories, names, note=""):
    """plot, show and save plot comparing entropies of multtiple runs

    Args:
        pop_histories (list): list of lists of 2d np arrays. one entry per run
        names (tuple): Names of runs.
        note (str, optional): short note, used for file path. Defaults to "".
    """
    for pop_hist in pop_histories:
        entropies = []
        for gen in pop_hist:
            # Per-gene entropy, i think that makes more sense.
            entropy = 0
            genes = np.transpose(gen)
            for gene in genes:
                try:
                    p1 = np.count_nonzero(gene)/len(gene)
                    p0 = 1 - p1
                    entropy -= log2(p1) * p1 + log2(p0) * p0
                except ValueError:
                    pass  # All instances of gene are the same, entropy 0
            entropies.append(entropy / gen.shape[1])

        plt.plot(entropies)
    plt.title("Entropy of runs - {:s}".format(note))
    plt.legend(names)
    plt.savefig(os.path.join("figs", "entropy_{:s}.png".format(note)))
    plt.show()
