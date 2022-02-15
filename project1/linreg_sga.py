import pandas as pd
# import matplotlib.pyplot as plt
import pickle

from SGA import sga, fitness_box_plot, entropy_plot
from LinReg import LinReg


def generate_fitness_func():
    """generates fitness function for selecting parameters
        and evaluating linreg on dataset

    Returns:
        function: The callable fitness function
    """
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
    # 50 gens, 200 pop takes 10 min
    hist, fitness = sga(generations=50, pop_size=200, bitstring_length=101,
                        fitness_func=fitness_func, mutation_chance=0.02,
                        maximize_fitness=False, breed_with_replacement=True,
                        crowding=False)
    # 50 gens, 200 pop takes 5 min
    hist_crowding, fitness_crowding = sga(
        generations=50, pop_size=200, bitstring_length=101,
        fitness_func=fitness_func, mutation_chance=0.02,
        maximize_fitness=False, crowding=True)
    # with open("linreg_hist.pickle", "rb") as fid:
    #     hist, fitness = pickle.load(fid)

    fitness_box_plot(fitness)
    fitness_box_plot(fitness_crowding)

    entropy_plot((hist, hist_crowding), ("SGA", "Deterministic crowding"),
                 note="crowding_analysis_linreg")

    with open("linreg_hist.pickle", 'wb') as fid:
        pickle.dump((hist, fitness), fid)

    with open("linreg_hist_crowding.pickle", 'wb') as fid:
        pickle.dump((hist_crowding, fitness_crowding), fid)
