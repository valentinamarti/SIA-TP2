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

def plot_diversity_avg_per_setup():
    """Genera un gráfico de diversidad promedio por setup (uno por archivo)."""
    for setup_name, setup_id in SETUPS:
        history_file = os.path.join(RESULTS_DIR, f"{setup_id}_history.csv")
        if not os.path.exists(history_file):
            print(f"[WARN] No existe {history_file}")
            continue

        df = pd.read_csv(history_file)

        # Agrupar por generación y calcular promedio y std de diversidad
        grouped = df.groupby("generation")["diversity"]
        mean_div = grouped.mean()
        std_div = grouped.std()

        generations = mean_div.index.values

        # Graficar curva + banda
        plt.figure(figsize=(8, 5))
        plt.plot(
            generations,
            mean_div,
            label="Diversidad promedio",
            color="blue",
            linewidth=2,
        )
        plt.fill_between(
            generations,
            mean_div - std_div,
            mean_div + std_div,
            alpha=0.2,
            color="blue",
        )

        plt.xlabel("Generación", fontsize=12)
        plt.ylabel("Diversidad (std fitness)", fontsize=12)
        plt.title(f"{setup_name} - Diversidad promedio", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        outdir = os.path.join(RESULTS_DIR, setup_id)
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, "diversity_avg.png")
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        plt.close()
        print(f"[OK] Guardado {outfile}")

def plot_diversity_std_per_setup():
    """Genera un gráfico de la diversidad promedio con banda de desvío por setup."""
    for setup_name, setup_id in SETUPS:
        history_file = os.path.join(RESULTS_DIR, f"{setup_id}_history.csv")
        if not os.path.exists(history_file):
            print(f"[WARN] No existe {history_file}")
            continue

        df = pd.read_csv(history_file)

        # Agrupar por generación y calcular promedio y std de diversidad
        grouped = df.groupby("generation")["diversity"]
        mean_div = grouped.mean()
        std_div = grouped.std()

        generations = mean_div.index.values

        # Graficar curva + banda de ±std
        plt.figure(figsize=(8, 5))
        plt.plot(
            generations,
            mean_div,
            label="Diversidad promedio",
            color="blue",
            linewidth=2,
        )
        plt.fill_between(
            generations,
            mean_div - std_div,
            mean_div + std_div,
            alpha=0.2,
            color="blue",
            label="±1 std"
        )

        plt.xlabel("Generación", fontsize=12)
        plt.ylabel("Diversidad (std fitness)", fontsize=12)
        plt.title(f"{setup_name} - Diversidad promedio con desviación", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        outdir = os.path.join(RESULTS_DIR, setup_id)
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, "diversity_std.png")
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        plt.close()
        print(f"[OK] Guardado {outfile}")

def plot_fitness_std_per_setup():
    """Genera un gráfico del fitness promedio con banda de desvío por setup."""
    for setup_name, setup_id in SETUPS:
        history_file = os.path.join(RESULTS_DIR, f"{setup_id}_history.csv")
        if not os.path.exists(history_file):
            print(f"[WARN] No existe {history_file}")
            continue

        df = pd.read_csv(history_file)

        # Agrupar por generación y calcular promedio y std del fitness
        grouped = df.groupby("generation")["fitness"]
        mean_fit = grouped.mean()
        std_fit = grouped.std()

        generations = mean_fit.index.values

        # Graficar curva + banda de ±std
        plt.figure(figsize=(8, 5))
        plt.plot(
            generations,
            mean_fit,
            label="Fitness promedio",
            color="orange",
            linewidth=2,
        )
        plt.fill_between(
            generations,
            mean_fit - std_fit,
            mean_fit + std_fit,
            alpha=0.2,
            color="orange",
            label="±1 std"
        )

        plt.xlabel("Generación", fontsize=12)
        plt.ylabel("Fitness (mejor individuo)", fontsize=12)
        plt.title(f"{setup_name} - Fitness promedio con desviación", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        outdir = os.path.join(RESULTS_DIR, setup_id)
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, "fitness_std.png")
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        plt.close()
        print(f"[OK] Guardado {outfile}")

def main():
    # 1) Fitness por corrida
    for setup_name, setup_id in SETUPS:
        plot_fitness(setup_name, setup_id)

    # 2) Fitness promedio + std
    plot_fitness_std_per_setup()

    # 3) Diversidad promedio + std
    plot_diversity_std_per_setup()

    # 4) Generaciones promedio (barras)
    plot_generations_bar()

    # 5) Tiempo promedio (barras)
    plot_time_bar()

if __name__ == "__main__":
    main()
