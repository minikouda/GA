from __future__ import annotations

import warnings
from typing import Any, Dict, Hashable, Optional, Tuple, TypedDict, Union, cast

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from .fitness import FitnessStrategy, get_fitness_strategy, set_global_n_workers
from .data_utils import preprocess_data

from . import operators

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
CacheKey = Tuple[Hashable, ...]
FitnessCache = Dict[CacheKey, Tuple[float, float]]


class SelectionResult(TypedDict):
    """Container for GA selection outputs."""
    selected: IntArray
    R2: float
    R2pen: float

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class GeneticSelector:
    """
    Genetic Algorithm for variable selection in Linear Regression.
    """
    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        penalty: float = 0.0,
        pop_size: int = 50,
        n_gen: int = 50,
        mutation_rate: float = 0.01,
        mutation_type: str = "bitflip",
        crossover_rate: float = 0.8,
        crossover_type: str = "uniform", 
        k_points: int = 2,

        n_splits: int = 10,
        random_state: Optional[int] = None,
        fitness_strategy: Union[str, FitnessStrategy] = "linear_regression",
        adaptive_mutation: bool = False,
        low_mutation_rate: Optional[float] = None,
        high_mutation_rate: Optional[float] = None,
        vectorized_ops: bool = False,
        patience: Optional[int] = 10,
        n_workers: int = 1,
    ) -> None:
        """
        Initialize the GeneticSelector.
        
        Args:
            X (np.ndarray): Predictor matrix.
            y (np.ndarray): Response vector.
            penalty (float): Penalty factor for the number of features.
            pop_size (int): Population size.
            n_gen (int): Number of generations.
            mutation_rate (float): Probability of mutation per gene.
            crossover_rate (float): Probability of crossover.
            n_splits (int): Number of CV splits.
            random_state (int, optional): Seed for reproducibility.
            fitness_strategy (str | FitnessStrategy): Registered name or callable used to compute cross-validated scores for each individual.
            adaptive_mutation (bool): Whether to use adaptive mutation based on fitness.
            low_mutation_rate (float, optional): Mutation rate for fit individuals.
            high_mutation_rate (float, optional): Mutation rate for unfit individuals.
            vectorized_ops (bool): Use vectorized selection/crossover/mutation (default False).
            patience (int, optional): Early-stopping patience in generations without improvement (default 10).
            n_workers (int): Number of workers used in cross-validation (-1 = all cores, 1 = sequential; default 1).
        """
        self.X: FloatArray = np.asarray(X, dtype=float)
        self.y: FloatArray = np.asarray(y, dtype=float)
        self.n_samples, self.n_features = self.X.shape
        self.penalty = penalty
        self.pop_size = pop_size
        self.n_gen = int(n_gen)
        self.mutation_rate = mutation_rate
        self.mutation_type = mutation_type.lower()
        self.crossover_rate = crossover_rate
        self.crossover_type = crossover_type
        self.k_points = k_points
        self.n_splits = n_splits
        self.rng: RandomState = np.random.RandomState(random_state)
        self.vectorized_ops = vectorized_ops
        self.n_workers = n_workers
        set_global_n_workers(self.n_workers)
        
        if isinstance(fitness_strategy, str):
            strategy_name = fitness_strategy.lower()
            self.fitness_strategy: FitnessStrategy = get_fitness_strategy(strategy_name)
            self._fitness_strategy_key: CacheKey = ("strategy", strategy_name)
            self.fitness_strategy_name = strategy_name
        else:
            self.fitness_strategy = fitness_strategy
            strategy_label = getattr(fitness_strategy, "__name__", "custom_fitness")
            self._fitness_strategy_key = ("callable", strategy_label, id(fitness_strategy))
            self.fitness_strategy_name = strategy_label

        # Cache for fitness evaluations: key = strategy identifier + tuple(selected_indices)
        self._fitness_cache: FitnessCache = {}
        
        # Ensure inputs are valid
        if self.X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if self.y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        # Initialize population: boolean matrix (pop_size, n_features)
        # We initialize with a random selection of features (approx 50% selected)
        self.population: IntArray = self.rng.randint(0, 2, size=(pop_size, self.n_features)).astype(np.int64)
        
        self.best_fitness: float = -np.inf
        self.best_individual: Optional[IntArray] = None
        self.best_r2: float = -np.inf
        self.patience = patience if patience is not None and patience > 0 else None
        self._no_improve_count = 0

        self.adaptive_mutation = adaptive_mutation
        if self.adaptive_mutation:
            # Default: fit individuals halved, unfit individuals doubled (with boundaries)
            self.low_rate = low_mutation_rate if low_mutation_rate is not None else max(1e-4, 0.5 * mutation_rate)
            self.high_rate = high_mutation_rate if high_mutation_rate is not None else min(0.1, 2 * mutation_rate)

    def fitness(self, individual: IntArray) -> Tuple[float, float]:
        """
        Calculate fitness for a single individual using a composite metric.

        The score combines:
         - mean cross-validated score from the configured strategy
         - sparsity penalty proportional to fraction of selected features

        Returns (fitness_score, mean_score). For regression with R^2, mean_score
        is 1 - SS_res/SS_tot as provided by sklearn's r2 scorer.
        """
        selected_indices_tuple = tuple(np.where(individual == 1)[0].tolist())

        # If no features are selected, return a very low fitness (keeps tests/behavior unchanged)
        if len(selected_indices_tuple) == 0:
            return -1.0, -1.0

        cache_key: CacheKey = self._fitness_strategy_key + selected_indices_tuple
        # Check cache to avoid repeated CV for same subset
        if cache_key in self._fitness_cache:
            return self._fitness_cache[cache_key]

        X_subset = self.X[:, list(selected_indices_tuple)]

        seed = int(self.rng.randint(0, 2**31 - 1))
        scores = self.fitness_strategy(X_subset, self.y, self.n_splits, seed)
        
        if scores.size == 0:
             return -1.0, -1.0

        mean_score = float(np.mean(scores))
        # Sparsity penalty proportional to fraction of selected features
        prop_selected = len(selected_indices_tuple) / float(self.n_features)
        fit_score = mean_score - (self.penalty * prop_selected)

        # Cache result
        self._fitness_cache[cache_key] = (fit_score, mean_score)

        return fit_score, mean_score

    def evaluate_population(self) -> Tuple[FloatArray, FloatArray]:
        """
        Evaluate fitness for the entire population.
        """
        fitness_scores: FloatArray = np.empty(self.pop_size, dtype=float)
        r2_scores: FloatArray = np.empty(self.pop_size, dtype=float)

        for i, ind in enumerate(self.population):
            fit, r2 = self.fitness(ind)
            fitness_scores[i] = fit
            r2_scores[i] = r2

        return fitness_scores, r2_scores

    def select_parents(self, fitness_scores: FloatArray) -> IntArray:
        """
        Select parents using Tournament Selection (vectorized optional).
        """
        if not self.vectorized_ops:
            parents: IntArray = np.empty_like(self.population)
            k = 3
            for i in range(self.pop_size):
                candidates_indices = self.rng.randint(0, self.pop_size, k)
                candidates_fitness = fitness_scores[candidates_indices]
                winner_idx = candidates_indices[np.argmax(candidates_fitness)]
                parents[i] = self.population[winner_idx]
            return parents

        k = 3
        candidates_indices = self.rng.randint(0, self.pop_size, size=(self.pop_size, k))
        candidates_fitness = fitness_scores[candidates_indices]
        winner_local_indices = np.argmax(candidates_fitness, axis=1, keepdims=True)
        winner_indices = np.take_along_axis(candidates_indices, winner_local_indices, axis=1).ravel()
        return self.population[winner_indices].copy()


    def crossover(self, parents: IntArray) -> IntArray:
        """
        Perform Crossover by calling the appropriate external method based on type.
        """
        if self.crossover_type == "uniform":
            return operators.uniform_crossover(
                parents, self.pop_size, self.n_features, self.crossover_rate, self.rng
            )
        elif self.crossover_type == "single":
            return operators.single_point_crossover(
                parents, self.pop_size, self.n_features, self.crossover_rate, self.rng
            )
        elif self.crossover_type == "kpoints":
            return operators.k_point_crossover(
                parents, self.pop_size, self.n_features, self.crossover_rate, self.k_points, self.rng
            )
        else:
             warnings.warn(f"Unknown crossover type '{self.crossover_type}'. Falling back to 'uniform'.")
             return operators.uniform_crossover(
                parents, self.pop_size, self.n_features, self.crossover_rate, self.rng
            )
    

    def mutate(self, offspring: IntArray, current_fitness: FloatArray, avg_fitness: float) -> IntArray:
        """
        Perform Mutation based on the configured mutation_type and adaptive setting.

        Args:
            offspring (IntArray): The offspring population.
            current_fitness (FloatArray): The fitness scores of the individuals (used for adaptive logic).
            avg_fitness (float): The average fitness of the current population.
        """
        # --- Bit-Flip Mutation ---
        if self.mutation_type == "bitflip":
            
            if not self.adaptive_mutation:
                # 1. Standard Bit-Flip (Calls external vectorized function)
                return operators.standard_bit_flip_mutation(offspring, self.mutation_rate, self.rng)
            else:
                # 2. Adaptive Bit-Flip (Per-individual rate, requires loop, calls external helper)
                for i in range(self.pop_size):
                    individual = offspring[i]
                    
                    if current_fitness[i] < avg_fitness:
                        # Low fitness -> High rate (Exploration)
                        rate = self.high_rate
                    else:
                        # High fitness -> Low rate (Exploitation)
                        rate = self.low_rate
                    
                    # Call external helper for individual mutation
                    offspring[i] = operators.bit_flip_mutation_individual(
                        individual, self.n_features, rate, self.rng
                    )
                return offspring

        # --- Swap Mutation ---
        elif self.mutation_type == "swap":
            if self.adaptive_mutation:
                 warnings.warn("Adaptive mutation is not supported for 'swap' type. Using standard mutation rate.")
                 
            return operators.swap_mutation(
                offspring, self.pop_size, self.n_features, self.mutation_rate, self.rng
            )
        
        # --- Unknown Type ---
        else:
            warnings.warn(f"Unknown mutation type '{self.mutation_type}'. Falling back to 'bitflip' (standard rate).")
            return operators.standard_bit_flip_mutation(offspring, self.mutation_rate, self.rng)
        
    def run(self) -> Tuple[IntArray, float, float]:
        """
        Run the Genetic Algorithm.
        """
        for gen in range(self.n_gen):
            fitness_scores, r2_scores = self.evaluate_population()
            
            avg_fitness = float(np.mean(fitness_scores)) # ensure type is float
            
            # Update best solution found so far
            max_fit_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fit_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[max_fit_idx]
                self.best_individual = self.population[max_fit_idx].copy()
                self.best_r2 = r2_scores[max_fit_idx]
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
            
            # Selection
            parents = self.select_parents(fitness_scores)
            
            # Crossover
            offspring = self.crossover(parents)
            
            # Mutation
            self.population = self.mutate(offspring, fitness_scores, avg_fitness)
            
            # Elitism: Let's replace the first individual with the best found so far (Elitism)
            if self.best_individual is not None:
                 self.population[0] = self.best_individual.copy()

            # Early stopping if no improvement for `patience` generations
            if self.patience is not None and self._no_improve_count >= self.patience:
                break

        if self.best_individual is None:
            raise RuntimeError("No feasible individual found during GA run.")

        # Final local search polishing can materially improve fitness for small problems.
        refined_ind, refined_r2, refined_fit = operators.local_refinement(
            self.best_individual,
            self.n_features,
            self.fitness,
        )
        if refined_fit > self.best_fitness:
            self.best_individual = refined_ind
            self.best_fitness = refined_fit
            self.best_r2 = refined_r2

        return self.best_individual, float(self.best_r2), float(self.best_fitness)

def select(
    X: ArrayLike,
    y: ArrayLike,
    penalty: float = 0.0,
    pop_size: int = 50,
    n_gen: int = 50,
    crossover_type: str = "uniform",
    mutation_type: str = "bitflip",
    **kwargs: Any
) -> SelectionResult:
    """Select variables using a Genetic Algorithm."""

    X_processed, y_processed = preprocess_data(X, y)

    if y_processed is None:
        raise ValueError("Response vector y could not be determined from the provided inputs.")

    X_array: FloatArray = np.asarray(X_processed, dtype=float)
    y_array: FloatArray = np.asarray(y_processed, dtype=float)

    # Extract kwargs or use defaults (no legacy aliases)
    mutation_rate = kwargs.pop('mutation_rate', 0.01)
    mutation_type = kwargs.pop('mutation_type', mutation_type)
    crossover_rate = kwargs.pop('crossover_rate', 0.8)
    crossover_type = kwargs.pop('crossover_type', crossover_type)
    n_splits = kwargs.pop('n_splits', 10)
    random_state = kwargs.pop('random_state', None)
    fitness_strategy = kwargs.pop('fitness_strategy', 'linear_regression')
    adaptive_mutation = kwargs.pop('adaptive_mutation', False)
    low_mutation_rate = kwargs.pop('low_mutation_rate', None)
    high_mutation_rate = kwargs.pop('high_mutation_rate', None)
    vectorized_ops = kwargs.pop('vectorized_ops', True)
    patience = kwargs.pop('patience', 10)
    k_points = kwargs.pop('k_points', 2)
    n_workers = kwargs.pop('n_workers', 1)

    if isinstance(patience, int) and patience <= 0:
        patience = None

    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"select() got unexpected keyword arguments: {unknown}")

    ga = GeneticSelector(
        X=X_array,
        y=y_array,
        penalty=penalty,
        pop_size=pop_size,
        n_gen=n_gen,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        n_splits=n_splits,
        random_state=random_state,
        fitness_strategy=fitness_strategy,
        adaptive_mutation=adaptive_mutation,
        low_mutation_rate=low_mutation_rate,
        high_mutation_rate=high_mutation_rate,
        vectorized_ops=vectorized_ops,
        patience=patience,
        k_points=k_points,
        n_workers=n_workers,
    )
    best_ind, best_r2, best_fit = ga.run()

    selected_indices: IntArray = np.where(best_ind == 1)[0].astype(np.int64)

    return SelectionResult(
        selected=selected_indices,
        R2=best_r2,
        R2pen=best_fit,
    )
