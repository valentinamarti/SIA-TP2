import random

import numpy as np

from utils.polygon import create_random_individual


def roulette(individuals, selected_k):
    """
        Roulette wheel selection (fitness-proportionate).

        Args:
            individuals (list[Individual]): population with precomputed fitness.
            selected_k (int): number of individuals to select.
        Returns:
            list[Individual]: list of selected individuals (duplicates possible).
    """
    # Step 1: Compute total fitness and relative probabilities
    total_fitness = sum(individual.fitness for individual in individuals)
    probabilities = [individual.fitness / total_fitness for individual in individuals]


    # Step 2: Compute cumulative relative probabilities
    cumulative_probabilities = []
    cumulative = 0.0
    for p in probabilities:
        cumulative += p
        cumulative_probabilities.append(cumulative)


    # Step 3: Select k individuals
    selected_individuals = []
    for n in range(selected_k):
        r = random.random()

        for i, q in enumerate(cumulative_probabilities):
            lower = cumulative_probabilities[i - 1] if i > 0 else 0.0
            if lower < r <= q:
                selected_individuals.append(individuals[i])
                break

    return selected_individuals

