import numpy as np
import pandas as pd
# from .LinReg import LinReg

data_df = pd.read_csv("data.csv")
data = data_df.values
print(data.shape)
n_features = data.shape[1]


def generate_pop(n_individuals, bitstring_length) -> list:
    pop = []
    for _ in range(n_individuals):
        pop.append(np.random.randint(2, size=bitstring_length))
    return pop


def parent_selection(pop, select_frac, fitness_func, maximize=True):
    """select parents from pop based on fitness_func

    Args:
        pop (list): population, list of bitstrings
        select_frac (float): fraction of population to select as parents (0, 1)
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

    # select select_frac parents from pop


def sine(x):
    n_bits = len(x)
    

if __name__ == '__main__':
    pop = generate_pop(20, 4)
    fitness_func = lambda x: 