from random import random

from selections.elite import elite
from utils.polygon import Individual


def replace_population_traditional(current_population, offspring, fitness, n_selected):
    """
        Generate the next generation by selecting n_selected individuals from
        the union of current_population + offspring.

        Args:
            current_population (list[Individual]): current generation
            offspring (list[Individual]): new individuals generated via crossovers/mutation
            fitness (function): function to evaluate fitness of an individual
            n_selected (int): number of individuals to keep for next generation

        Returns:
            list[Individual]: next generation of size n_selected
    """
    combined_population = offspring + current_population
    sorted_combined_population = sorted(combined_population, key=fitness, reverse=True)

    return sorted_combined_population[:n_selected]
