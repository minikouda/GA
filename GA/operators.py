import warnings
from typing import Callable, Tuple
import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray

FloatArray = NDArray[np.float64] 
IntArray = NDArray[np.int64]

# --- Crossover Operators ---

def uniform_crossover(parents: IntArray, pop_size: int, n_features: int, crossover_rate: float, rng: RandomState) -> IntArray:
    """Mask-based Crossover (Uniform Crossover)."""
    offspring = np.empty_like(parents)
    num_pairs = pop_size // 2

    for idx in range(num_pairs):
        i = 2 * idx
        p1 = parents[i]
        p2 = parents[i + 1] if i + 1 < pop_size else parents[0]

        if rng.random() < crossover_rate:
            mask: NDArray[np.bool_] = rng.randint(0, 2, size=n_features).astype(bool)

            o1 = np.empty_like(p1)
            o2 = np.empty_like(p2)
            
            o1[mask] = p1[mask]
            o1[~mask] = p2[~mask]
            o2[mask] = p2[mask]
            o2[~mask] = p1[~mask]

            offspring[i] = o1
            if i + 1 < pop_size:
                offspring[i+1] = o2
        else:
            offspring[i] = p1
            if i + 1 < pop_size:
                offspring[i+1] = p2
    return offspring

def single_point_crossover(parents: IntArray, pop_size: int, n_features: int, crossover_rate: float, rng: RandomState) -> IntArray:
    """Single-Point Crossover."""
    offspring = np.empty_like(parents)
    num_pairs = pop_size // 2

    for idx in range(num_pairs):
        i = 2 * idx
        p1 = parents[i]
        p2 = parents[i + 1] if i + 1 < pop_size else parents[0]
        
        if rng.random() < crossover_rate:
            cut_point = rng.randint(1, n_features)
            
            o1 = np.empty_like(p1)
            o2 = np.empty_like(p2)

            o1[:cut_point] = p1[:cut_point]
            o1[cut_point:] = p2[cut_point:]
            
            o2[:cut_point] = p2[:cut_point]
            o2[cut_point:] = p1[cut_point:]

            offspring[i] = o1
            if i + 1 < pop_size:
                offspring[i+1] = o2
        else:
            offspring[i] = p1
            if i + 1 < pop_size:
                offspring[i+1] = p2
    return offspring

def k_point_crossover(parents: IntArray, pop_size: int, n_features: int, crossover_rate: float, k_points: int, rng: RandomState) -> IntArray:
    """K-Point Crossover (Generalized Crossover)."""
    if k_points <= 0 or k_points >= n_features:
        warnings.warn(f"k_points={k_points} is invalid. Falling back to Uniform Crossover.")
        return uniform_crossover(parents, pop_size, n_features, crossover_rate, rng)

    offspring = np.empty_like(parents)
    num_pairs = pop_size // 2

    for idx in range(num_pairs):
        i = 2 * idx
        p1 = parents[i]
        p2 = parents[i + 1] if i + 1 < pop_size else parents[0]
        
        if rng.random() < crossover_rate:
            
            cut_points = np.sort(rng.choice(n_features, size=k_points, replace=False))
            
            o1 = np.empty_like(p1)
            o2 = np.empty_like(p2)
            boundaries = np.concatenate(([0], cut_points, [n_features]))
            
            for j in range(len(boundaries) - 1):
                start = boundaries[j]
                end = boundaries[j+1]

                if j % 2 == 0:
                    o1[start:end] = p1[start:end]
                    o2[start:end] = p2[start:end]
                else:
                    o1[start:end] = p2[start:end]
                    o2[start:end] = p1[start:end]

            offspring[i] = o1
            if i + 1 < pop_size:
                offspring[i+1] = o2
        else:
            offspring[i] = p1
            if i + 1 < pop_size:
                offspring[i+1] = p2
    return offspring

# --- Mutation Operators ---

def bit_flip_mutation_individual(individual: IntArray, n_features: int, rate: float, rng: RandomState) -> IntArray:
    """Helper function for bit-flip on a single individual (used by Adaptive Mutation)."""
    mutation_mask = rng.rand(n_features) < rate
    individual[mutation_mask] = 1 - individual[mutation_mask]
    return individual

def swap_mutation(offspring: IntArray, pop_size: int, n_features: int, mutation_rate: float, rng: RandomState) -> IntArray:
    """Perform Swap Mutation on the population with a rate per individual."""
    for i in range(pop_size):
        if rng.random() < mutation_rate:
            if n_features >= 2:
                idx1, idx2 = rng.choice(n_features, size=2, replace=False)
                # Swap elements
                offspring[i, idx1], offspring[i, idx2] = offspring[i, idx2], offspring[i, idx1]
    return offspring

def standard_bit_flip_mutation(offspring: IntArray, mutation_rate: float, rng: RandomState) -> IntArray:
    """Standard (vectorized) Bit-Flip Mutation."""
    mutation_mask = rng.rand(*offspring.shape) < mutation_rate
    offspring[mutation_mask] = 1 - offspring[mutation_mask]
    return offspring

# --- Local Refinement (Hill-Climb) ---

def local_refinement(
    individual: IntArray,
    n_features: int,
    fitness_fn: Callable[[IntArray], Tuple[float, float]],
    max_checks_cap: int = 64,
) -> Tuple[IntArray, float, float]:
    """Greedy single-bit hill-climb on the given solution."""
    best = individual.copy()
    best_fit, best_score = fitness_fn(best)
    max_checks = min(n_features, max_checks_cap)

    for idx in range(max_checks):
        candidate = best.copy()
        candidate[idx] = 1 - candidate[idx]
        fit, score = fitness_fn(candidate)
        if fit > best_fit:
            best = candidate
            best_fit = fit
            best_score = score

    return best, best_score, best_fit