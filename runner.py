import time
import os
import csv
from main import run_ga, load_image

def run_experiment(name, params, summary_file, history_file, runs=5):
    """Corre un experimento N veces y guarda resultados en dos CSV (final + historial)."""
    print(f"\n=== Ejecutando {name} ===")
    os.makedirs("results", exist_ok=True)

    # Inicializar archivos
    write_header_summary = not os.path.exists(summary_file)
    write_header_history = not os.path.exists(history_file)

    with open(summary_file, mode="a", newline="\n", encoding="utf-8") as f_summary, \
         open(history_file, mode="a", newline="\n", encoding="utf-8") as f_history:

        # CSV para resumen final
        summary_writer = csv.DictWriter(
            f_summary,
            fieldnames=["setup", "run_id", "fitness_final", "generations", "time_sec"]
        )
        if write_header_summary:
            summary_writer.writeheader()

        # CSV para historial por generación
        history_writer = csv.DictWriter(
            f_history,
            fieldnames=["setup", "run_id", "generation", "fitness", "diversity"]
        )
        if write_header_history:
            history_writer.writeheader()

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
                target_error=params["target_error"],
                generation_amount=params["generation_amount"],
            )
            end = time.time()
            elapsed = end - start

            # Guardar resumen
            summary_writer.writerow({
                "setup": name,
                "run_id": i + 1,
                "fitness_final": result["best"].fitness,
                "generations": result["generations"],
                "time_sec": elapsed
            })

            # Guardar historial completo
            for gen, (fit, div) in enumerate(zip(result["fitness_history"], result["diversity_history"])):
                history_writer.writerow({
                    "setup": name,
                    "run_id": i + 1,
                    "generation": gen,
                    "fitness": fit,
                    "diversity": div
                })

            print(f"  Run {i+1}/{runs} guardado en {summary_file} y {history_file}")


def main():
    base_params = {
        "image": "images/blue_square.png",
        "size": (128, 128),
        "population_size": 40,
        "max_polygons": 40,
        "polygon_sides": 3,
        "mutate_structure": False,
        #"target_error": 0.05,
        "target_error": 0.03,
        "generation_amount": 4000,
    }

    # Experimento 1: Baseline clásico
    exp1 = {**base_params, "selection": "roulette", "crossover": "one_point",
            "mutation": "gen", "replacement": "traditional"}
    run_experiment("Experimento 1: Baseline clásico", exp1,
                   "results/exp1.csv", "results/exp1_history.csv")

    # Experimento 2: Exploración fuerte
    exp2 = {**base_params, "selection": "boltzmann", "crossover": "uniform_crossover",
            "mutation": "uniform_multi_gen", "replacement": "youth_bias"}
    run_experiment("Experimento 2: Exploración fuerte", exp2,
                   "results/exp2.csv", "results/exp2_history.csv")

    # Experimento 3: Presión alta
    exp3 = {**base_params, "selection": "tournament_deterministic", "crossover": "two_point",
            "mutation": "limited_multi_gen", "replacement": "traditional"}
    run_experiment("Experimento 3: Presión alta", exp3,
                   "results/exp3.csv", "results/exp3_history.csv")

    # Experimento 4: Balanceado
    exp4 = {**base_params, "selection": "ranking", "crossover": "anular",
            "mutation": "uniform_multi_gen", "replacement": "youth_bias"}
    run_experiment("Experimento 4: Balanceado", exp4,
                   "results/exp4.csv", "results/exp4_history.csv")


if __name__ == "__main__":
    main()
