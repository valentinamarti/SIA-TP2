import random
import numpy as np

def elite(population, num_parents):
    return

def roulette(population, num_parents):
    return

def universal(population, num_parents):
    return

def boltzmann(population, num_parents, temperature=1.0):
    return

def tournament_deterministic(population, num_parents, k=3):
    return

def tournament_probabilistic(population, num_parents, k=3, p=0.75):
    return

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