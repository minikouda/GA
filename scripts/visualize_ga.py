"""
Animate the Genetic Algorithm's evolution and save as a GIF.

Panels (side-by-side):
  Left   – Population chromosomes heatmap (rows = individuals, cols = features)
  Center – Best fitness curve up to current generation
  Right  – Feature selection frequency bar chart

Usage:
    python scripts/visualize_ga.py [--output ga_animation.gif] [--fps 3]
                                   [--pop-size 30] [--n-gen 40]
                                   [--n-features 20] [--random-state 42]
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from GA.GA import GeneticSelector


# ---------------------------------------------------------------------------
# Instrumented subclass that records population history per generation
# ---------------------------------------------------------------------------

class TrackingGeneticSelector(GeneticSelector):
    """GA that stores a snapshot of the population after each generation."""

    def run(self):
        self.history_populations = []   # list of (pop_size, n_features) int arrays
        self.history_best_fitness = []  # best fitness seen so far at each generation
        self.history_gen_best = []      # best fitness in this generation

        for gen in range(self.n_gen):
            fitness_scores, r2_scores = self.evaluate_population()
            avg_fitness = float(np.mean(fitness_scores))

            max_fit_idx = int(np.argmax(fitness_scores))
            gen_best = float(fitness_scores[max_fit_idx])

            if gen_best > self.best_fitness:
                self.best_fitness = gen_best
                self.best_individual = self.population[max_fit_idx].copy()
                self.best_r2 = float(r2_scores[max_fit_idx])
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1

            # Record *before* evolving so frame shows what was evaluated
            self.history_populations.append(self.population.copy())
            self.history_best_fitness.append(self.best_fitness)
            self.history_gen_best.append(gen_best)

            parents = self.select_parents(fitness_scores)
            offspring = self.crossover(parents)
            self.population = self.mutate(offspring, fitness_scores, avg_fitness)

            if self.best_individual is not None:
                self.population[0] = self.best_individual.copy()

            if self.patience is not None and self._no_improve_count >= self.patience:
                break

        if self.best_individual is None:
            raise RuntimeError("No feasible individual found during GA run.")

        return self.best_individual, float(self.best_r2), float(self.best_fitness)


# ---------------------------------------------------------------------------
# Build a synthetic regression problem
# ---------------------------------------------------------------------------

def make_synthetic_data(n_samples=200, n_features=20, n_informative=6, random_state=42):
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    idx = rng.choice(n_features, n_informative, replace=False)
    true_coef[idx] = rng.randn(n_informative) * 2
    y = X @ true_coef + rng.randn(n_samples) * 0.5
    feature_names = [f"X{i+1}" for i in range(n_features)]
    return X, y, feature_names, idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Animate GA evolution to GIF.")
    parser.add_argument("--output", default="ga_animation.gif")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--n-gen", type=int, default=15)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-informative", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--penalty", type=float, default=0.05)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    args = parser.parse_args()

    print("Generating synthetic dataset …")
    X, y, feat_names, true_idx = make_synthetic_data(
        n_features=args.n_features,
        n_informative=args.n_informative,
        random_state=args.random_state,
    )

    print("Running GA (tracking history) …")
    ga = TrackingGeneticSelector(
        X=X,
        y=y,
        penalty=args.penalty,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        mutation_rate=args.mutation_rate,
        crossover_rate=0.8,
        n_splits=5,
        random_state=args.random_state,
        patience=None,
    )
    best_ind, best_r2, best_fit = ga.run()
    n_frames = len(ga.history_populations)
    print(f"  Ran {n_frames} generations  |  best R²={best_r2:.4f}  |  best fitness={best_fit:.4f}")

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 5), facecolor="#0d1117")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax_pop  = fig.add_subplot(gs[0])   # population heatmap
    ax_fit  = fig.add_subplot(gs[1])   # fitness curve
    ax_freq = fig.add_subplot(gs[2])   # feature selection frequency

    for ax in (ax_pop, ax_fit, ax_freq):
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#e6edf3")

    # ---- Panel 1: Population heatmap ----
    pop0 = ga.history_populations[0].astype(float)
    im = ax_pop.imshow(
        pop0, aspect="auto", interpolation="nearest",
        cmap="YlOrRd", vmin=0, vmax=1,
    )
    ax_pop.set_xlabel("Feature index")
    ax_pop.set_ylabel("Individual")
    ax_pop.set_title("Population chromosomes")
    ax_pop.set_xticks(range(args.n_features))
    ax_pop.set_xticklabels(feat_names, rotation=90, fontsize=6)
    # Highlight true informative features with a green rectangle
    for fi in true_idx:
        ax_pop.axvline(x=fi, color="#3fb950", alpha=0.4, lw=6)
    # Colorbar legend for heatmap
    from matplotlib.patches import Patch
    pop_legend_els = [
        Patch(facecolor="#ffffb2", label="not selected"),
        Patch(facecolor="#d7191c", label="selected"),
        Patch(facecolor="#3fb950", alpha=0.4, label="true signal col"),
    ]
    ax_pop.legend(
        handles=pop_legend_els, fontsize=11,
        facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e",
        loc="upper right",
    )
    gen_text = ax_pop.text(
        0.02, 1.03, "Gen 1", transform=ax_pop.transAxes,
        color="#58a6ff", fontsize=9, va="bottom",
    )

    # ---- Panel 2: Fitness curve ----
    ax_fit.set_xlim(0, n_frames)
    ymin = min(ga.history_gen_best) * 0.95
    ymax = max(ga.history_best_fitness) * 1.05
    ax_fit.set_ylim(ymin, ymax)
    ax_fit.set_xlabel("Generation")
    ax_fit.set_ylabel("Fitness (penalized R²)")
    ax_fit.set_title("Best fitness over time")

    line_best, = ax_fit.plot([], [], color="#58a6ff", lw=2, label="best-so-far")
    line_gen,  = ax_fit.plot([], [], color="#f78166", lw=1.2, ls="--", alpha=0.7, label="gen best")
    ax_fit.legend(fontsize=11, facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e")

    # ---- Panel 3: Feature frequency bar chart ----
    freq0 = ga.history_populations[0].mean(axis=0)
    bars = ax_freq.bar(
        range(args.n_features), freq0,
        color="#388bfd", edgecolor="none",
    )
    # True feature markers
    for fi in true_idx:
        bars[fi].set_color("#3fb950")
    ax_freq.set_ylim(0, 1)
    ax_freq.set_xlabel("Feature index")
    ax_freq.set_ylabel("Selection frequency")
    ax_freq.set_title("Feature selection frequency")
    ax_freq.set_xticks(range(args.n_features))
    ax_freq.set_xticklabels(feat_names, rotation=90, fontsize=6)
    legend_els = [
        Patch(facecolor="#388bfd", label="other"),
        Patch(facecolor="#3fb950", label="true signal"),
    ]
    ax_freq.legend(
        handles=legend_els, fontsize=11,
        facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e",
        loc="upper right",
    )

    # ------------------------------------------------------------------
    # Animation update
    # ------------------------------------------------------------------
    def update(frame):
        pop = ga.history_populations[frame]

        # Heatmap
        im.set_data(pop.astype(float))
        gen_text.set_text(f"Gen {frame + 1}")

        # Fitness lines
        gens = list(range(frame + 1))
        line_best.set_data(gens, ga.history_best_fitness[:frame + 1])
        line_gen.set_data(gens, ga.history_gen_best[:frame + 1])

        # Frequency bars
        freq = pop.mean(axis=0)
        for i, bar in enumerate(bars):
            bar.set_height(freq[i])
            bar.set_color("#3fb950" if i in true_idx else "#388bfd")

        return im, gen_text, line_best, line_gen, *bars

    ani = FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // args.fps, blit=True,
    )

    out_path = args.output
    print(f"Saving animation to {out_path} …")
    writer = PillowWriter(fps=args.fps)
    ani.save(out_path, writer=writer, dpi=130)
    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
