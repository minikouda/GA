import argparse
import os
import sys

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GA import select
from GA.fitness import list_available_strategies
from GA.data_utils import load_data
from sklearn.datasets import load_breast_cancer, load_diabetes


CLASSIFICATION_METRIC_TO_STRATEGY = {
    "accuracy": "logistic_regression_accuracy",
    "precision": "logistic_regression_precision",
    "cross_entropy": "logistic_regression_log_loss",
    "recall": "logistic_regression_recall",
}

REGRESSION_METRIC_TO_STRATEGY = {
    "r2": "linear_regression",
}

DEFAULT_METRIC_BY_TASK = {
    "classification": "accuracy",
    "regression": "r2",
}

METRIC_SUFFIX_MAP = {
    "cross_entropy": "log_loss",
}

AVAILABLE_STRATEGIES = set(list_available_strategies())

parser = argparse.ArgumentParser(description="Run Genetic Algorithm for feature selection.")
parser.add_argument("--fitness", help="Fitness strategy to use (e.g., 'linear_regression', 'logistic_regression', 'svr').")
parser.add_argument("--penalty", type=float, default=0.0, help="Penalty factor for the number of features.")
parser.add_argument("--pop-size", dest="pop_size", type=int, default=50, help="Population size for the genetic algorithm.")
parser.add_argument("--n-gen", dest="n_gen", type=int, default=20, help="Number of generations to run the genetic algorithm.")
parser.add_argument("--mutation-rate", type=float, default=0.01, help="Probability of mutation per gene.")
parser.add_argument("--crossover-rate", type=float, default=0.8, help="Probability of crossover.")
parser.add_argument("--crossover-type", type=str, default="uniform", choices=["uniform", "single", "kpoints"], help="Type of crossover operator.")
parser.add_argument("--k-points", type=int, default=2, help="Number of cut points for k-point crossover (used when --crossover-type=kpoints)")
parser.add_argument("--n-splits", type=int, default=10, help="Number of cross-validation splits.")
parser.add_argument("--vectorized", action="store_true", help="Enable vectorized GA operators for speed.")
parser.add_argument("--adaptive-mutation", action="store_true", help="Use adaptive mutation rates based on population fitness.")
parser.add_argument("--low-mutation-rate", type=float, help="Low mutation rate used when individuals perform well (requires --adaptive-mutation).")
parser.add_argument("--high-mutation-rate", type=float, help="High mutation rate used when individuals underperform (requires --adaptive-mutation).")
parser.add_argument("--patience", type=int, default=10, help="Generations without improvement before early stopping (<=0 disables).")
parser.add_argument("--n-workers", dest="n_workers", type=int, default=1, help="Number of workers for cross-validation (-1 uses all cores).")
parser.add_argument("--data-path", type=str, help="Path to a data file (CSV, Excel, TXT). The last column will be used as the target variable (y).")
parser.add_argument("--task-type", choices=["classification", "regression"], help="Problem type. Defaults based on metric or regression if unspecified.")
parser.add_argument("--metric", choices=["cross_entropy", "accuracy", "precision", "recall", "r2"], help="Scoring metric used for model evaluation.")
parser.add_argument("--random-state", type=int, help="Seed used for reproducible GA runs and model evaluation randomness.")
args = parser.parse_args()


def _infer_task_from_metric(metric: str | None) -> str | None:
    if metric is None:
        return None
    if metric in CLASSIFICATION_METRIC_TO_STRATEGY:
        return "classification"
    if metric in REGRESSION_METRIC_TO_STRATEGY:
        return "regression"
    return None


def _metric_suffix(metric: str) -> str:
    return METRIC_SUFFIX_MAP.get(metric, metric)


def _resolve_configuration() -> tuple[str, str | None, str]:
    metric = args.metric.lower() if args.metric else None
    task = args.task_type.lower() if args.task_type else None
    fitness = args.fitness.lower() if args.fitness else None

    inferred_task = _infer_task_from_metric(metric)

    if fitness:
        if metric and inferred_task is None:
            raise ValueError(f"Metric '{metric}' is not supported.")
        if task and inferred_task and task != inferred_task:
            raise ValueError(f"Metric '{metric}' is not valid for task '{task}'.")
        if task is None:
            task = inferred_task or "regression"

        # Normalize base fitness names to task-specific strategies
        base = fitness
        if base in {"random_forest", "rf"}:
            fitness = "random_forest_classification" if task == "classification" else "random_forest_regression"
        elif base in {"gradient_boosting", "gb"}:
            fitness = "gradient_boosting_classification" if task == "classification" else "gradient_boosting_regression"
        elif base in {"logistic", "logistic_regression"}:
            fitness = "logistic_regression"
        elif base in {"linear", "linear_regression", "ols"}:
            fitness = "linear_regression"

        if metric:
            suffix = _metric_suffix(metric)
            candidate = f"{fitness}_{suffix}"
            if candidate in AVAILABLE_STRATEGIES:
                return task, metric, candidate

        if fitness not in AVAILABLE_STRATEGIES:
            raise ValueError(f"Fitness strategy '{fitness}' is not registered.")

        return task, metric, fitness

    if task is None:
        task = inferred_task or "regression"

    if metric is None:
        metric = DEFAULT_METRIC_BY_TASK[task]

    metric_map = CLASSIFICATION_METRIC_TO_STRATEGY if task == "classification" else REGRESSION_METRIC_TO_STRATEGY
    if metric not in metric_map:
        valid = ", ".join(sorted(metric_map))
        raise ValueError(f"Metric '{metric}' is not supported for task '{task}'. Choose from: {valid}.")

    return task, metric, metric_map[metric]


try:
    task_type, metric_name, fitness_strategy = _resolve_configuration()
except ValueError as err:
    print(f"Configuration error: {err}")
    sys.exit(1)

if args.data_path:
    try:
        # Use load_data to support various file formats (CSV, Excel, TXT)
        data_obj = load_data(args.data_path)

        if isinstance(data_obj, pd.DataFrame):
            X = data_obj.iloc[:, :-1].to_numpy()
            y_series = data_obj.iloc[:, -1]
            if pd_types.is_numeric_dtype(y_series):
                y = y_series.to_numpy()
            else:
                y = pd.factorize(y_series)[0]
        elif isinstance(data_obj, pd.Series):
            raise ValueError("Expected feature columns alongside target; received a single Series.")
        else:
            data_array = np.asarray(data_obj)
            if data_array.ndim != 2 or data_array.shape[1] < 2:
                raise ValueError("Data array must be 2D with at least two columns (features + target).")
            X = data_array[:, :-1]
            y_raw = data_array[:, -1]
            y_array = np.asarray(y_raw)
            if y_array.dtype.kind not in "biufc":
                y = pd.factorize(pd.Series(y_array))[0]
            else:
                y = y_array

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        print(f"Loaded dataset from {args.data_path} with {X.shape[0]} samples and {X.shape[1]} features.")
    except Exception as e:
        print(f"Error loading data from {args.data_path}: {e}")
        exit(1)
else:
    if task_type == "classification":
        X, y = load_breast_cancer(return_X_y=True)
        print("Using default 'breast_cancer' dataset from sklearn.")
    else:
        X, y = load_diabetes(return_X_y=True)
        print("Using default 'diabetes' dataset from sklearn.")

X_input = np.asarray(X, dtype=float)
y_input = np.asarray(y)

result = select(
    X_input,
    y_input,
    penalty=args.penalty,
    pop_size=args.pop_size,
    n_gen=args.n_gen,
    fitness_strategy=fitness_strategy,
    mutation_rate=args.mutation_rate,
    crossover_rate=args.crossover_rate,
    crossover_type=args.crossover_type,
    k_points=args.k_points,
    n_splits=args.n_splits,
    vectorized_ops=args.vectorized,
    adaptive_mutation=args.adaptive_mutation,
    low_mutation_rate=args.low_mutation_rate,
    high_mutation_rate=args.high_mutation_rate,
    patience=args.patience,
    n_workers=args.n_workers,
    random_state=args.random_state,
)
print("GA Selection Result:")
print(f"  Selected Features Indices: {result['selected']}")
metric_label = (metric_name or "custom metric").replace("_", " ").title()
print(f"  Cross-validated Score ({metric_label}): {result['R2']:.4f}")
print(f"  Penalized Fitness Score: {result['R2pen']:.4f}")