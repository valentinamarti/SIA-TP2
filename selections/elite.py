import math
import random

from utils.polygon import Individual, create_random_individual


def elite(individuals, selected_k):
    """
    Elite selection using n(i) = ceil((k - i)/n)

    Args:
        individuals (list[Individual]): the population of individuals.
        selected_k (int): total number of individuals to select.
    Returns:
        list[Individual]: list of selected individuals.
    """
    n = len(individuals)

    sorted_individuals = sorted(individuals, key=lambda individual: individual.fitness, reverse=True)
    selected_individuals = []

    for i, ind in enumerate(sorted_individuals):
        n_copies = math.ceil((selected_k - i) / n)
        selected_individuals.extend([ind] * n_copies)

    return selected_individuals[:selected_k]