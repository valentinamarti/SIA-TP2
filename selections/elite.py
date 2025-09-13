import math

from utils.polygon import Individual


def elite(individuals, selected_k, fitness):
    """
    Elite selection using n(i) = ceil((k - i)/n)

    Args:
        individuals (list[Individual]): the population of individuals.
        selected_k (int): total number of individuals to select.
        fitness (function): fitness function that takes an individual and returns its fitness value.

    Returns:
        list[Individual]: list of selected individuals.
    """
    n = len(individuals)

    sorted_individuals = sorted(individuals, key=fitness, reverse=True)
    selected_individuals = []

    for i, ind in enumerate(sorted_individuals):
        n_copies = math.ceil((selected_k - i) / n)
        selected_individuals.extend([ind] * n_copies)

    return selected_individuals[:selected_k]

