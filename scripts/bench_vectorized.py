#!/usr/bin/env python3
"""Benchmark GA vectorized vs standard operators on synthetic regression data.

This enhanced version also reports a per-phase timing breakdown to make it clear
which operations are vectorized and where time is spent each generation:
 - Fitness evaluation (K-fold CV via the configured fitness strategy)
 - Selection (tournament)  [vectorized when --vectorized mode is enabled]
 - Crossover (uniform/single/k-point)
 - Mutation (bit-flip is vectorized; swap uses a loop)
 - Local refinement (post-run greedy hill-climb)

Note: Vectorization flag primarily changes selection (see GA.GeneticSelector.select_parents).
Bit-flip mutation is already vectorized in both modes.
"""
import time
import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GA import select
from GA.GA import GeneticSelector  # direct import for instrumentation
from GA.operators import local_refinement

parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=2000)
parser.add_argument("--n-features", type=int, default=200)
parser.add_argument("--pop-size", dest="pop_size", type=int, default=100)
parser.add_argument("--n-gen", dest="n_gen", type=int, default=40)
parser.add_argument("--penalty", type=float, default=0.005)
parser.add_argument("--strategy", type=str, default="linear_regression")
parser.add_argument("--n-splits", type=int, default=5)
parser.add_argument("--random-state", type=int, default=123)
parser.add_argument("--crossover-type", type=str, default="kpoints", choices=["uniform", "single", "kpoints"]) 
parser.add_argument("--k-points", type=int, default=3, help="k for k-point crossover (when --crossover-type=kpoints)")
parser.add_argument("--mutation-type", type=str, default="swap", choices=["bitflip", "swap"]) 
parser.add_argument("--n-workers", dest="n_workers", type=int, default=-1, help="Workers for CV (-1 uses all cores)")
parser.add_argument("--no-breakdown", action="store_true", help="Skip per-phase breakdown (keeps legacy behavior)")
args = parser.parse_args()

rng = np.random.RandomState(args.random_state)
X = rng.normal(size=(args.n_samples, args.n_features))
# Create continuous target with signal from a small feature subset
signal_idx = rng.choice(args.n_features, size=6, replace=False)
weights = rng.normal(size=6)
y = X[:, signal_idx].dot(weights) + 0.5 * rng.normal(size=args.n_samples)

def run_instrumented(vectorized_ops: bool):
    """Run GA with per-phase timing using GeneticSelector's public methods.

    Returns a dict with result fields and a timings dict.
    """
    ga = GeneticSelector(
        X=X,
        y=y,
        penalty=args.penalty,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        mutation_rate=0.01,
        mutation_type=args.mutation_type,
        crossover_rate=0.8,
        crossover_type=args.crossover_type,
        k_points=args.k_points,
        n_splits=args.n_splits,
        random_state=args.random_state,
        fitness_strategy=args.strategy,
        adaptive_mutation=False,
        vectorized_ops=vectorized_ops,
        patience=10,
        n_workers=args.n_workers,
    )

    t_eval = 0.0
    t_select = 0.0
    t_cross = 0.0
    t_mut = 0.0
    t_refine = 0.0

    t_total_start = time.time()
    for gen in range(ga.n_gen):
        t0 = time.time()
        fitness_scores, r2_scores = ga.evaluate_population()
        t_eval += time.time() - t0

        avg_fitness = float(np.mean(fitness_scores))
        max_fit_idx = int(np.argmax(fitness_scores))
        if fitness_scores[max_fit_idx] > ga.best_fitness:
            ga.best_fitness = float(fitness_scores[max_fit_idx])
            ga.best_individual = ga.population[max_fit_idx].copy()
            ga.best_r2 = float(r2_scores[max_fit_idx])
            ga._no_improve_count = 0
        else:
            ga._no_improve_count += 1

        t0 = time.time()
        parents = ga.select_parents(fitness_scores)
        t_select += time.time() - t0

        t0 = time.time()
        offspring = ga.crossover(parents)
        t_cross += time.time() - t0

        t0 = time.time()
        ga.population = ga.mutate(offspring, fitness_scores, avg_fitness)
        t_mut += time.time() - t0

        if ga.best_individual is not None:
            ga.population[0] = ga.best_individual.copy()

        if ga.patience is not None and ga._no_improve_count >= ga.patience:
            break

    if ga.best_individual is None:
        raise RuntimeError("No feasible individual found during GA run.")

    t0 = time.time()
    refined_ind, refined_r2, refined_fit = local_refinement(ga.best_individual, ga.n_features, ga.fitness)
    t_refine += time.time() - t0
    if refined_fit > ga.best_fitness:
        ga.best_individual = refined_ind
        ga.best_fitness = float(refined_fit)
        ga.best_r2 = float(refined_r2)

    total_time = time.time() - t_total_start

    selected = np.where(ga.best_individual == 1)[0]
    return {
        "result": {"selected": selected, "R2": ga.best_r2, "R2pen": ga.best_fitness},
        "timings": {
            "total": total_time,
            "evaluate": t_eval,
            "selection": t_select,
            "crossover": t_cross,
            "mutation": t_mut,
            "local_refine": t_refine,
            "generations_run": gen + 1,
        },
        "cache": {
            "hits": getattr(ga, "_cache_hits", 0),
            "misses": getattr(ga, "_cache_misses", 0),
            "unique_subsets": len(getattr(ga, "_fitness_cache", {})),
        },
    }

# Standard (non-vectorized)
std = run_instrumented(vectorized_ops=False)

# Vectorized
vec = run_instrumented(vectorized_ops=True)

print("=== Benchmark results ===")
print(f"Data: n_samples={args.n_samples}, n_features={args.n_features}")
print(f"GA: pop_size={args.pop_size}, n_gen={args.n_gen}, penalty={args.penalty}")
print(f"Strategy: {args.strategy}; Crossover={args.crossover_type} (k={args.k_points}); Mutation={args.mutation_type}; n_workers={args.n_workers}")

res_std = std["result"]
res_vec = vec["result"]
std_time = std["timings"]["total"]
vec_time = vec["timings"]["total"]

print("-- Standard --")
print(f"Time: {std_time:.2f}s | Score: {res_std['R2']:.4f} | Penalized: {res_std['R2pen']:.4f} | Selected {len(res_std['selected'])}")
if not args.no_breakdown:
    t = std["timings"]
    total = t["total"] or 1.0
    print("  Breakdown:")
    print(f"   - Evaluate (fitness/CV): {t['evaluate']:.2f}s ({100*t['evaluate']/total:.1f}%)")
    print(f"   - Selection (tournament): {t['selection']:.2f}s ({100*t['selection']/total:.1f}%)")
    print(f"   - Crossover: {t['crossover']:.2f}s ({100*t['crossover']/total:.1f}%)")
    print(f"   - Mutation: {t['mutation']:.2f}s ({100*t['mutation']/total:.1f}%)")
    print(f"   - Local refine: {t['local_refine']:.2f}s ({100*t['local_refine']/total:.1f}%)")
    print(f"   - Generations run: {t['generations_run']}")
    c = std.get("cache", {})
    if c:
        total_lookups = c["hits"] + c["misses"] or 1
        hit_rate = 100.0 * c["hits"] / total_lookups
        print(f"   - Cache: hits={c['hits']}, misses={c['misses']} (hit rate {hit_rate:.1f}%), unique subsets={c['unique_subsets']}")

print("-- Vectorized --")
print(f"Time: {vec_time:.2f}s | Score: {res_vec['R2']:.4f} | Penalized: {res_vec['R2pen']:.4f} | Selected {len(res_vec['selected'])}")
if not args.no_breakdown:
    t = vec["timings"]
    total = t["total"] or 1.0
    print("  Breakdown:")
    print(f"   - Evaluate (fitness/CV): {t['evaluate']:.2f}s ({100*t['evaluate']/total:.1f}%)")
    print(f"   - Selection (tournament): {t['selection']:.2f}s ({100*t['selection']/total:.1f}%) [vectorized]")
    print(f"   - Crossover: {t['crossover']:.2f}s ({100*t['crossover']/total:.1f}%)")
    print(f"   - Mutation: {t['mutation']:.2f}s ({100*t['mutation']/total:.1f}%) (bit-flip is vectorized)")
    print(f"   - Local refine: {t['local_refine']:.2f}s ({100*t['local_refine']/total:.1f}%)")
    print(f"   - Generations run: {t['generations_run']}")
    c = vec.get("cache", {})
    if c:
        total_lookups = c["hits"] + c["misses"] or 1
        hit_rate = 100.0 * c["hits"] / total_lookups
        print(f"   - Cache: hits={c['hits']}, misses={c['misses']} (hit rate {hit_rate:.1f}%), unique subsets={c['unique_subsets']}")

speedup = std_time / vec_time if vec_time > 0 else float('inf')
print(f"Speedup (std/vec): {speedup:.2f}x")
