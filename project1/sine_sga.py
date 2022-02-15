import os
import numpy as np
import matplotlib.pyplot as plt
import sys

from SGA import sga, fitness_box_plot, entropy_plot
from math import sin


def bitstring_to_num(x) -> float:
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


def sine(x) -> float:
    """Converts bitstring x into number in range [0, 128]
        and returns sine of that number

    Args:
        x (numpy array): bitstring

    Returns:
        float: sine of bitstring
    """
    return sin(bitstring_to_num(x))


def sine_restricted(x) -> float:
    real = bitstring_to_num(x)
    if 5 <= real <= 10:
        return sin(real)
    return -2


def plot_sine_pop(pop, generation, note="") -> None:

    x = [bitstring_to_num(ind) for ind in pop]
    y = [sine(n) for n in pop]
    sine_x = np.linspace(0, 128, 1000)
    plt.plot(sine_x, [sin(a) for a in sine_x], "r-")
    plt.plot(x, y, "o")
    plt.xlim(0, 128)
    plt.ylim(-1.2, 1.2)
    plt.xticks(list(range(0, 129, 16)))
    plt.yticks([-1, 0, 1])
    plt.title("Generation {:03d} - {:s}".format(generation, note))
    plt.savefig(os.path.join(
        "figs", "sine_{:s}_gen_{:03d}.png".format(note, generation)))
    plt.close()


if __name__ == '__main__':

    if len(sys.argv) == 1:
        fitness_func = sine

    elif sys.argv[1] == "restricted":
        print("Running while restricting solution to [5, 10]")
        fitness_func = sine_restricted
    else:
        exit(1)
    hist, fitness = sga(100, 1000, 50, fitness_func, 0.005,
                        maximize_fitness=True, crowding=False)
    hist_crowding, fitness_crowding = sga(100, 1000, 50, fitness_func, 0.008,
                                          maximize_fitness=True, crowding=True)
    for gen in [0, 1, 10, 50, 99]:
        plot_sine_pop(hist[gen], gen, note="no_crowd")
        plot_sine_pop(hist_crowding[gen], gen, note="crowding")

    fitness_box_plot(fitness_crowding)
    fitness_box_plot(fitness)

    entropy_plot((hist, hist_crowding), names=(
        "no crowding", "crowding"), note="crowding_analysis")
