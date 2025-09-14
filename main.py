from typing import Dict, Any, List
import numpy as np

from src.crossover_methods import one_point_crossover, uniform_crossover
from src.generations_methods import replace_population_traditional, replace_population_young_bias
from src.mutation_methods import mutate_individual
from src.selection_methods import select_individuals
from utils.draw import save_rendered
from utils.image import load_image
from utils.polygon import create_random_individual, Individual
from src.fitness import FitnessEvaluator
GENERATION_AMOUNT = 100000


def run_ga(image: np.ndarray,
           polygon_sides: int = 3,
           selection_method: str = "tournament",
           mutation_method: str = "uniform_multi_gen",
           mutate_structure: bool = False,
           crossover: str = "one_point",
           population_size: int = 60,
           replacement_method: str = "traditional",
           max_polygons: int = None) -> Dict[str, Any]:
    size = (image.shape[1], image.shape[0])  # (width, height)
    num_polygons = 10 if max_polygons is None else max_polygons
    population: List[Individual] = [
        create_random_individual(num_polygons, polygon_sides)
        for _ in range(population_size)
    ]
    fitness_evaluator = FitnessEvaluator(image)
    fitness_evaluator.evaluate_population(population)

    for gen in range(GENERATION_AMOUNT):
        new_population: List[Individual] = []

        fitness_evaluator.evaluate_population(population)

        # selections
        parents = select_individuals(selection_method, population, population_size)

        # crossover
        for i in range(0, len(parents), 2):
            p1, p2 = parents[i], parents[i + 1]
            if np.random.rand() < 0.8:  # crossover probability
                if crossover == "one_point":
                    c1, c2 = one_point_crossover(p1, p2)
                elif crossover == "uniform":
                    c1, c2 = uniform_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()  # sin crossover, se copian tal cual

        # mutate individuals
        for ind in parents:
            mutated = mutate_individual(mutation_method, ind, size, prob=0.2, structural=mutate_structure)
            new_population.append(mutated)

        # new generations
        if replacement_method == "traditional":
            population = replace_population_traditional(population, new_population, population_size)
        elif replacement_method == "youth_bias":
            population = replace_population_young_bias(population, new_population, population_size)

        # print(f"Generación {gen + 1}: {len(population)} individuos")

    best = population[0]
    save_rendered(best, size,filename = "results/output_rgb.png",)
    print(f"Fitness: {best.fitness}\nPolígonos: {len(best.polygons)}")
    return {"best": best}


if __name__ == "__main__":
    target_img = load_image("images/argentina.png", size=(128,128))
    result = run_ga(target_img,
                    max_polygons=40,
                    population_size=40,
                    mutate_structure=True,
                    mutation_method="uniform_multi_gen",
                    selection_method="tournament",
                    crossover="one_point",
                    replacement_method="traditional",
                    polygon_sides=3)


