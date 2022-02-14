import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import pickle

from SGA import sga, generate_pop, fitness_bar_plot
from LinReg import LinReg


def generate_fitness_func():

    linreg = LinReg()
    data_df = pd.read_csv("data.csv")
    data = data_df.values[:, :-1]
    y = data_df.values[:, -1]

    def feature_select_and_fit(bitstring) -> float:
        selected_data = linreg.get_columns(data, bitstring)
        error = linreg.get_fitness(selected_data, y)
        return error

    return feature_select_and_fit


if __name__ == '__main__':
    fitness_func = generate_fitness_func()
    test_strings = generate_pop(3, 101)
    for bitstring in test_strings:
        print(bitstring)
        print(fitness_func(bitstring))
    hist, fitness = sga(50, 200, 101, fitness_func, mutation_chance=0.01,
                        maximize_fitness=False, breed_with_replacement=True)

    # with open("linreg_hist.pickle", "rb") as fid:
    #     hist, fitness = pickle.load(fid)

    fitness_bar_plot(fitness)

    # for gen in [0, 1, 10, 49]:
    #     print(fitness[gen])
    #     print(np.array([np.sum(bitstring) for bitstring in hist[gen]]))
    #     print()

    with open("linreg_hist.pickle", 'wb') as fid:
        pickle.dump((hist, fitness), fid)
