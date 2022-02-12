import os
import numpy as np
import matplotlib.pyplot as plt
import sys

from SGA import sga
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


def plot_sine_pop(pop, generation) -> None:

    x = [bitstring_to_num(ind) for ind in pop]
    y = [sine(n) for n in pop]
    sine_x = np.linspace(0, 128, 1000)
    plt.plot(sine_x, [sin(a) for a in sine_x], "r-")
    plt.plot(x, y, "o")
    plt.xlim(0, 128)
    plt.ylim(-1.2, 1.2)
    plt.xticks(list(range(0, 129, 16)))
    plt.yticks([-1, 0, 1])
    plt.title("Generation {:03d}".format(generation))
    plt.savefig(os.path.join("figs", "sine_gen_{:03d}.png".format(generation)))
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        hist, fitness = sga(500, 1000, 50, sine, 0.005)
    elif sys.argv[1] == "restricted":
        print("Running while restricting solution to [5, 10]")
        hist, fitness = sga(100, 1000, 50, sine_restricted, 0.005)
    else:
        exit(1)

    for gen in [0, 1, 10, 99]:
        print(fitness[gen])
        plot_sine_pop(hist[gen], gen)
