import math
import random
import numpy as np

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

def _select_from_cumulative(individuals, pointers, aptitude_function):
    """Auxiliary: given pointers and fitness, return selected individuals."""
    # Step 1: Compute total fitness and relative probabilities
    total_aptitude = sum(aptitude_function(ind) for ind in individuals)
    probabilities = [aptitude_function(ind) / total_aptitude for ind in individuals]

    # Step 2: Compute cumulative relative probabilities
    cumulative_probabilities = []
    cumulative = 0.0
    for p in probabilities:
        cumulative += p
        cumulative_probabilities.append(cumulative)

    # Step 3: Select k individuals
    selected = []
    for r in pointers:
        for i, q in enumerate(cumulative_probabilities):
            lower = cumulative_probabilities[i - 1] if i > 0 else 0.0
            if lower < r <= q:
                selected.append(individuals[i])
                break
    return selected


def roulette(individuals, selected_k, aptitude_function = lambda ind: ind.fitness):
    """
        Roulette wheel selection (fitness-proportionate).

        Args:
            aptitude_function: function to compute the aptitude of an individual.
            individuals (list[Individual]): population with precomputed fitness.
            selected_k (int): number of individuals to select.
        Returns:
            list[Individual]: list of selected individuals (duplicates possible).
    """
    pointers = sorted(random.random() for _ in range(selected_k))
    return _select_from_cumulative(individuals, pointers, aptitude_function)


def universal(individuals, selected_k, aptitude_function=lambda ind: ind.fitness):
    """
       Universal Selection
       Like roulette selection but generates k evenly spaced pointers.

       Args:
           individuals (list[Individual]): population with precomputed fitness.
           selected_k (int): number of individuals to select.
       Returns:
           list[Individual]: list of selected individuals (duplicates possible).
    """
    r = random.random() / selected_k
    pointers = [r + j / selected_k for j in range(selected_k)]
    return _select_from_cumulative(individuals, pointers, aptitude_function)


def boltzmann(population, num_parents, temperature=1.0):
    """    
    Fórmula: ExpVal(i, g, T) = (e^(f(i)/T)) / (<e^(f(x)/T)>g)
    """
    if not population:
        return []
    
    exp_fitness = [math.exp(individual.fitness / temperature) for individual in population]
    
    avg_exp_fitness = np.mean(exp_fitness)
    
    def boltzmann_aptitude(individual):
        exp_val = math.exp(individual.fitness / temperature)
        return exp_val / avg_exp_fitness
    
    return roulette(population, num_parents, boltzmann_aptitude)

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
        probs = [0.7, 0.2, 0.1] 

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
    """
    Fórmula: f'(i) = (N - rank(i)) / N
    - N: tamaño de la población
    - rank(i): ranking del individuo (1 = mejor fitness, N = peor fitness)
    """
    if not population:
        return []
    
    N = len(population)
    
    # Ordenar población por fitness (mayor a menor)
    sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    
    def ranking_aptitude(individual):
        # Encontrar el ranking del individuo (1 = mejor, N = peor)
        rank = sorted_population.index(individual) + 1
        # Aplicar fórmula: f'(i) = (N - rank(i)) / N
        return (N - rank) / N
    
    return roulette(population, num_parents, ranking_aptitude)


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