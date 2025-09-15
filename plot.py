import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

SETUPS = [
    ("Experimento 1: Baseline clásico", "exp1"),
    ("Experimento 2: Exploración fuerte", "exp2"),
    ("Experimento 3: Presión alta", "exp3"),
    ("Experimento 4: Balanceado", "exp4"),
]

RESULTS_DIR = "results"


def plot_fitness_diversity(setup_name, setup_id):
    """Genera gráficos de fitness + diversidad para cada corrida de un setup."""
    history_file = os.path.join(RESULTS_DIR, f"{setup_id}_history.csv")
    if not os.path.exists(history_file):
        print(f"[WARN] No existe {history_file}")
        return

    df = pd.read_csv(history_file)

    for run_id, run_data in df.groupby("run_id"):
        outdir = os.path.join(RESULTS_DIR, setup_id, f"graph_run_{run_id}")
        os.makedirs(outdir, exist_ok=True)

        plt.figure()
        generations = run_data["generation"].values
        fitness = run_data["fitness"].values
        diversity = run_data["diversity"].values

        plt.plot(generations, fitness, label="Fitness (mejor)", color="blue")
        plt.plot(generations, diversity, label="Diversidad (std fitness)", color="red", linestyle="--")

        plt.xlabel("Generación")
        plt.ylabel("Valor")
        plt.title(f"{setup_name} - Run {run_id}")
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        outfile = os.path.join(outdir, "fitness_diversity.png")
        plt.savefig(outfile)
        plt.close()
        print(f"[OK] Guardado {outfile}")


def plot_generations_bar():
    """Gráfico de barras: generaciones promedio con std."""
    data = []
    for setup_name, setup_id in SETUPS:
        summary_file = os.path.join(RESULTS_DIR, f"{setup_id}.csv")
        if not os.path.exists(summary_file):
            continue
        df = pd.read_csv(summary_file)
        gens = df["generations"].values
        mean, std = np.mean(gens), np.std(gens)
        data.append((setup_name, mean, std))

    if not data:
        return

    labels, means, stds = zip(*data)
    x = np.arange(len(labels))

    plt.figure()
    plt.bar(x, means, yerr=stds, capsize=5, color="skyblue")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Generaciones promedio")
    plt.title("Generaciones promedio por setup (±1 std)")

    outdir = os.path.join(RESULTS_DIR, "summary")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "generations.png")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    print(f"[OK] Guardado {outfile}")


def plot_time_bar():
    """Gráfico de barras: tiempo promedio con std."""
    data = []
    for setup_name, setup_id in SETUPS:
        summary_file = os.path.join(RESULTS_DIR, f"{setup_id}.csv")
        if not os.path.exists(summary_file):
            continue
        df = pd.read_csv(summary_file)
        times = df["time_sec"].values
        mean, std = np.mean(times), np.std(times)
        data.append((setup_name, mean, std))

    if not data:
        return

    labels, means, stds = zip(*data)
    x = np.arange(len(labels))

    plt.figure()
    plt.bar(x, means, yerr=stds, capsize=5, color="lightgreen")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Tiempo promedio (s)")
    plt.title("Tiempo promedio por setup (±1 std)")

    outdir = os.path.join(RESULTS_DIR, "summary")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "time.png")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    print(f"[OK] Guardado {outfile}")


def main():
    # Graficos 1: fitness + diversidad
    for setup_name, setup_id in SETUPS:
        plot_fitness_diversity(setup_name, setup_id)

    # Graficos 2: generaciones promedio
    plot_generations_bar()

    # Graficos 3: tiempo promedio
    plot_time_bar()


if __name__ == "__main__":
    main()
