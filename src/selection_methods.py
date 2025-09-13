import math
import random

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


def universal(population, num_parents):
    return

def boltzmann(population, num_parents, temperature=1.0):
    return

import random

def tournament_deterministic(population, num_parents, k=3):
    """Deterministic tournament selection."""
    parents = []
    for _ in range(num_parents):
        tournament = random.sample(population, k)  # elijo k individuos al azar
        winner = min(tournament, key=lambda ind: ind.fitness)  # mejor fitness
        parents.append(winner)
    return parents


def tournament_probabilistic(population, num_parents, k=3, probs=None):
    """Probabilistic tournament selection."""
    if probs is None:
        probs = [0.7, 0.2, 0.1] # 70%, 20%, 10%

    parents = []
    for _ in range(num_parents):
        tournament = random.sample(population, k)
        tournament_sorted = sorted(tournament, key=lambda ind: ind.fitness)

        probs_adj = probs[:k]
        s = sum(probs_adj)
        probs_adj = [p / s for p in probs_adj]

        winner = random.choices(tournament_sorted, weights=probs_adj, k=1)[0]
        parents.append(winner)
    return parents


def tournament(population, num_parents, deterministic=True):
    if deterministic:
        return tournament_deterministic(population, num_parents)
    else:
        return tournament_probabilistic(population, num_parents)

def ranking(population, num_parents):
    return


def select_individuals(selection_method: str, population: list, num_parents: int, **kwargs):
    return SELECTION_METHODS[selection_method](population, num_parents, **kwargs)


SELECTION_METHODS = {
    "elite": elite,
    "roulette": roulette,
    "universal": universal,
    "boltzmann": boltzmann,
    "tournament": tournament,
    "ranking": ranking
}