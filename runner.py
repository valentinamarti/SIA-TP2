import time
import os
from main import run_ga, save_metrics_csv, load_image

def run_experiment(name, params, output_file, runs=5):
    """Corre un experimento N veces y guarda resultados en CSV"""
    print(f"\n=== Ejecutando {name} ===")
    os.makedirs("results", exist_ok=True)

    for i in range(runs):
        target_img = load_image(params["image"], size=params["size"])
        start = time.time()
        result = run_ga(
            target_img,
            max_polygons=params["max_polygons"],
            population_size=params["population_size"],
            polygon_sides=params["polygon_sides"],
            selection_method=params["selection"],
            crossover=params["crossover"],
            mutation_method=params["mutation"],
            replacement_method=params["replacement"],
            mutate_structure=params["mutate_structure"],
            target_error=0.03,
            generation_amount=params["generation_amount"],
        )
        end = time.time()
        elapsed = end - start

        metrics = {
            "fitness": result["best"].fitness,
            "polygons": len(result["best"].polygons),
            "generations": result["generations"],
            "time_sec": elapsed,
            "run_id": i + 1
        }
        save_metrics_csv(params, metrics, filename=output_file)

        print(f"  Run {i+1}/{runs} guardado en {output_file}")


def main():
    base_params = {
        "image": "images/blue_square.png",
        "size": (128, 128),
        "population_size": 40,
        "max_polygons": 40,
        "polygon_sides": 3,
        "mutate_structure": False,
        "target_error": 0.02,
        "generation_amount": 100000,
    }

    # Experimento 1: Baseline clásico
    exp1 = {**base_params, "selection": "roulette", "crossover": "one_point",
            "mutation": "gen", "replacement": "traditional"}
    run_experiment("Experimento 1: Baseline clásico", exp1, "results/exp1.csv")

    # Experimento 2: Exploración fuerte
    exp2 = {**base_params, "selection": "boltzmann", "crossover": "uniform_crossover",
            "mutation": "uniform_multi_gen", "replacement": "youth_bias"}
    run_experiment("Experimento 2: Exploración fuerte", exp2, "results/exp2.csv")

    # Experimento 3: Presión alta
    exp3 = {**base_params, "selection": "tournament_deterministic", "crossover": "two_point",
            "mutation": "limited_multi_gen", "replacement": "traditional"}
    run_experiment("Experimento 3: Presión alta", exp3, "results/exp3.csv")

    # Experimento 4: Balanceado
    exp4 = {**base_params, "selection": "ranking", "crossover": "anular",
            "mutation": "uniform_multi_gen", "replacement": "youth_bias"}
    run_experiment("Experimento 4: Balanceado", exp4, "results/exp4.csv")


if __name__ == "__main__":
    main()
