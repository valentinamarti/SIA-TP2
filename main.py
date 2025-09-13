import random
from typing import Dict, Any, List
import numpy as np
from src.mutation_methods import mutate_individual
from src.selection_methods import select_individuals
from utils.draw import save_rendered
from utils.image import load_image
from utils.polygon import create_random_individual, Individual

GENERATION_AMOUNT = 200


def run_ga(image: np.ndarray,
           polygon_sides: int = 3,
           selection_method: str = "tournament",
           mutation_method: str = "gen",
           crossover: str = "one_point",
           population_size: int = 40,
           max_polygons: int = None) -> Dict[str, Any]:
    size = (image.shape[1], image.shape[0])  # (width, height)
    num_polygons = 10 if max_polygons is None else max_polygons
    population: List[Individual] = [
        create_random_individual(num_polygons, polygon_sides)
        for _ in range(population_size)
    ]

    for gen in range(GENERATION_AMOUNT): # 200 generations
        new_population: List[Individual] = []
        # TODO: crossover

        # mutate individuals
        for ind in population:
            mutated = mutate_individual(mutation_method, ind, size, prob=0.2)
            new_population.append(mutated)

        # TODO: calculate new generation's fitness
        # Selección
        # population = select_individuals(selection_method, new_population, population_size)
        population = new_population
        print(f"Generación {gen + 1}: {len(population)} individuos")

    best = population[0]
    save_rendered(best, size)
    return {"best": best}


if __name__ == "__main__":
    target_img = load_image("images/blue_square.png", size=(128,128))
    result = run_ga(target_img)
