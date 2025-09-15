import os

import numpy as np
import time
import argparse
import cv2
import csv

from typing import Dict, Any, List
from src.crossover_methods import one_point_crossover, uniform_crossover
from src.generations_methods import replace_population_traditional, replace_population_young_bias
from src.mutation_methods import mutate_individual
from src.selection_methods import select_individuals
from src.fitness import FitnessEvaluator
from utils.draw import save_rendered
from utils.image import load_image
from utils.polygon import create_random_individual, Individual

def save_metrics_csv(params: dict, metrics: dict, filename="results/output.csv"):
    """
    Guarda los parámetros y métricas de una corrida en un CSV.
    Si el archivo no existe, escribe encabezado.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Unir parámetros y métricas en un solo dict para la fila
    row = {**params, **metrics}

    write_header = not os.path.exists(filename)

    with open(filename, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)



def list_poligons(best: Individual, size: tuple[int, int]):
    w, h = size
    print("\nPoligonos del mejor individuo:")
    for i, poly in enumerate(best.polygons, start=1):
        # Escalar vértices de [0,1] → pixeles
        puntos = [(int(x * w), int(y * h)) for x, y in poly.vertices]
        # Reordenar color de BGRA → RGBA
        b, g, r, a = poly.color
        color = (r, g, b, a)
        puntos_str = ", ".join(f"({x},{y})" for x, y in puntos)
        print(f"{i}. {puntos_str} - {color}")



def run_ga(image: np.ndarray,
           polygon_sides: int = 3,
           selection_method: str = "tournament",
           mutation_method: str = "uniform_multi_gen",
           mutate_structure: bool = False,
           crossover: str = "one_point",
           population_size: int = 60,
           replacement_method: str = "traditional",
           max_polygons: int = None,
           target_error: float | None = None,
           generation_amount: int = 100000) -> Dict[str, Any]:
    size = (image.shape[1], image.shape[0])  # (width, height)
    num_polygons = 10 if max_polygons is None else max_polygons
    population: List[Individual] = [
        create_random_individual(num_polygons, polygon_sides)
        for _ in range(population_size)
    ]
    fitness_evaluator = FitnessEvaluator(image)
    fitness_evaluator.evaluate_population(population)
    population = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    gen = 0
    while gen < generation_amount and (
        target_error is None or population[0].error is None or population[0].error > target_error
    ):
        # selections
        parents = select_individuals(selection_method, population, population_size)

        # crossover
        children = []
        for i in range(0, len(parents), 2):
            p1, p2 = parents[i], parents[i + 1]
            if np.random.rand() < 0.8:  # crossover probability
                if crossover == "one_point":
                    c1, c2 = one_point_crossover(p1, p2)
                else:  # crossover == "uniform"
                    c1, c2 = uniform_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()  # sin crossover
            children.extend([c1, c2])

        # mutate children
        new_population = []
        for child in children:
            mutated = mutate_individual(mutation_method, child, size, prob=0.2, structural=mutate_structure)
            new_population.append(mutated)

        # Evaluar fitness de los hijos antes del reemplazo
        fitness_evaluator.evaluate_population(new_population)

        # new generations
        if replacement_method == "traditional":
            population = replace_population_traditional(population, new_population, population_size)
        elif replacement_method == "youth_bias":
            population = replace_population_young_bias(population, new_population, population_size)

        # Reordenamos y avanzamos gen para condición del while
        fitness_evaluator.evaluate_population(population)
        gen += 1

        # print(f"Generación {gen + 1}: {len(population)} individuos")

    best = population[0]
    save_rendered(best, size,filename = "results/output_rgb.png",)
    #print(f"Fitness: {best.fitness}\nPolígonos: {len(best.polygons)}\nGen: {gen}")
    return {"best": best, "generations": gen}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm for image approximation")

    # parámetros CLI
    parser.add_argument("--image", type=str, default="images/blue_square.png", help="Ruta de la imagen objetivo")
    parser.add_argument("--size", type=int, nargs=2, default=(128, 128), help="Tamaño de la imagen (w h)")
    parser.add_argument("--max_polygons", type=int, default=40, help="Número máximo de polígonos")
    parser.add_argument("--population_size", type=int, default=40, help="Tamaño de la población")
    parser.add_argument("--mutate_structure", action="store_true", help="Permitir mutación estructural")
    parser.add_argument("--mutation_method", type=str, default="uniform_multi_gen", help="Método de mutación")
    parser.add_argument("--selection_method", type=str, default="boltzmann", help="Método de selección")
    parser.add_argument("--crossover", type=str, default="uniform_crossover", help="Método de cruce")
    parser.add_argument("--replacement_method", type=str, default="traditional", help="Método de reemplazo")
    parser.add_argument("--polygon_sides", type=int, default=3, help="Cantidad de lados por polígono")
    parser.add_argument("--target_error", type=float, default=0.01, help="Error objetivo")
    parser.add_argument("--generation_amount", type=int, default=100000, help="Número máximo de generaciones")
    parser.add_argument("--output_file", type=str, default="results/output.csv", help="Ruta al archivo CSV donde guardar métricas")

    args = parser.parse_args()

    target_img = load_image(args.image, size=tuple(args.size))

    print("\nEjecutando AG con los siguientes parámetros:")
    print(f" Imagen objetivo       : {args.image}")
    print(f" Tamaño                : {args.size}")
    print(f" Máx. polígonos        : {args.max_polygons}")
    print(f" Lados por polígono    : {args.polygon_sides}")
    print(f" Tamaño población      : {args.population_size}")
    print(f" Generaciones máx.     : {args.generation_amount}")
    print(f" Método de selección   : {args.selection_method}")
    print(f" Método de cruce       : {args.crossover}")
    print(f" Método de mutación    : {args.mutation_method}")
    print(f" Mutación estructural  : {args.mutate_structure}")
    print(f" Método de reemplazo   : {args.replacement_method}")
    print(f" Error objetivo        : {args.target_error}\n")

    start = time.time()
    result = run_ga(target_img,
                    max_polygons=args.max_polygons,
                    population_size=args.population_size,
                    mutate_structure=args.mutate_structure,
                    mutation_method=args.mutation_method,
                    selection_method=args.selection_method,
                    crossover=args.crossover,
                    replacement_method=args.replacement_method,
                    polygon_sides=args.polygon_sides,
                    target_error=args.target_error,
                    generation_amount=args.generation_amount)
    end = time.time()
    elapsed = end - start
    print(f"Tiempo total de ejecución: {elapsed:.2f} segundos")

    metrics = {
        "fitness": result["best"].fitness,
        "polygons": len(result["best"].polygons),
        "generations": result["generations"],
        "time_sec": elapsed
    }

    params = {
        "image": args.image,
        "size": args.size,
        "population_size": args.population_size,
        "max_polygons": args.max_polygons,
        "polygon_sides": args.polygon_sides,
        "selection": args.selection_method,
        "crossover": args.crossover,
        "mutation": args.mutation_method,
        "mutate_structure": args.mutate_structure,
        "replacement": args.replacement_method,
        "target_error": args.target_error,
        "generation_amount": args.generation_amount
    }

    save_metrics_csv(params, metrics, filename=args.output_file)
    print("Resultados guardados en {args.output_file}")

    # Output required
    list_poligons(result["best"], args.size)

    # Show image generated
    # img = cv2.imread("results/output_rgb.png")
    # cv2.imshow("Resultado GA", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()





