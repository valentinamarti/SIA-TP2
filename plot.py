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


def plot_fitness(setup_name, setup_id):
    """Genera gráficos de fitness por generación para cada corrida de un setup."""
    history_file = os.path.join(RESULTS_DIR, f"{setup_id}_history.csv")
    if not os.path.exists(history_file):
        print(f"[WARN] No existe {history_file}")
        return

    df = pd.read_csv(history_file)

    for run_id, run_data in df.groupby("run_id"):
        outdir = os.path.join(RESULTS_DIR, setup_id, f"graph_run_{run_id}")
        os.makedirs(outdir, exist_ok=True)

        generations = run_data["generation"].values
        fitness = run_data["fitness"].values

        plt.figure(figsize=(8, 5))
        plt.plot(
            generations,
            fitness,
            label="Fitness (mejor individuo)",
            color="orange",
            linewidth=2,
        )

        plt.xlabel("Generación", fontsize=12)
        plt.ylabel("Fitness del mejor individuo", fontsize=12)
        plt.title(f"{setup_name} - Run {run_id}", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        outfile = os.path.join(outdir, "fitness.png")
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
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

    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=5, color="skyblue")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Generaciones promedio")
    plt.title("Generaciones promedio por setup (±1 std)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)

    outdir = os.path.join(RESULTS_DIR, "summary")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "generations.png")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
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

    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=5, color="lightgreen")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Tiempo promedio (s)")
    plt.title("Tiempo promedio por setup (±1 std)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)

    outdir = os.path.join(RESULTS_DIR, "summary")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "time.png")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"[OK] Guardado {outfile}")


def main():
    # Graficos 1: fitness por corrida
    for setup_name, setup_id in SETUPS:
        plot_fitness(setup_name, setup_id)

    # Graficos 2: generaciones promedio
    plot_generations_bar()

    # Graficos 3: tiempo promedio
    plot_time_bar()


if __name__ == "__main__":
    main()
