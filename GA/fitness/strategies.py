"""Built-in fitness strategy implementations."""
from __future__ import annotations

from typing import Tuple, Dict, Any, Optional, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.svm import SVC, SVR
from sklearn.metrics import make_scorer

from .registry import FitnessScores, FitnessStrategy, register_fitness_strategy

FloatArray = NDArray[np.float64]

# Global configuration for parallelism in cross-validation. This is set from the
# higher-level GA API so that users can configure the number of workers
# (n_workers) without touching these internals.
GLOBAL_N_WORKERS: int = -1

# Optional per-strategy model kwargs, set via setter to allow tuning without API churn.
_STRATEGY_PARAMS: Dict[str, Dict[str, Any]] = {}

# Optional fixed CV objects to ensure identical splits across all subset evaluations in a run.
_USE_FIXED_SPLITS: bool = False
_FIXED_KF: Optional[KFold] = None
_FIXED_SKF: Optional[StratifiedKFold] = None


def set_global_n_workers(n_workers: int) -> None:
    """Set the global n_workers used in all fitness strategies' CV calls.

    Args:
        n_workers: Number of workers for scikit-learn's cross_val_score.
            -1 uses all available cores, 1 is fully sequential, >1 uses that
            many workers (subject to backend limits).
    """
    global GLOBAL_N_WORKERS
    GLOBAL_N_WORKERS = n_workers


def set_strategy_params(name: str, **kwargs: Any) -> None:
    """Set default estimator kwargs for a registered strategy.

    The mapping applies to strategies that construct estimators in this module.
    Examples:
        set_strategy_params("random_forest_regression", n_estimators=100, max_depth=10)
        set_strategy_params("logistic_regression", solver="lbfgs", max_iter=2000)
    """
    _STRATEGY_PARAMS[name.lower()] = dict(kwargs)


def enable_fixed_cv_splits(n_splits: int, seed: int) -> None:
    """Enable fixed CV splits for both KFold and StratifiedKFold with shuffle.

    When enabled, all strategies will reuse these CV objects instead of creating
    new ones per subset, improving determinism/comparability.
    """
    global _USE_FIXED_SPLITS, _FIXED_KF, _FIXED_SKF
    _USE_FIXED_SPLITS = True
    _FIXED_KF = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _FIXED_SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def disable_fixed_cv_splits() -> None:
    """Disable fixed CV splits, returning to per-call seeded fold creation."""
    global _USE_FIXED_SPLITS, _FIXED_KF, _FIXED_SKF
    _USE_FIXED_SPLITS = False
    _FIXED_KF = None
    _FIXED_SKF = None


def _run_cv(
    model,
    X_subset: FloatArray,
    y: FloatArray,
    *,
    n_splits: int,
    seed: int,
    scoring: Any,
    stratified: bool = False,
) -> FitnessScores:
    if X_subset.size == 0:
        return np.full(n_splits, -1.0)

    if stratified:
        cv = _FIXED_SKF if _USE_FIXED_SPLITS and _FIXED_SKF is not None else StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
    else:
        cv = _FIXED_KF if _USE_FIXED_SPLITS and _FIXED_KF is not None else KFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
    try:
        scores = cross_val_score(
            model,
            X_subset,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=GLOBAL_N_WORKERS,
        )
    except ValueError as e:
        # Common stratification failure due to class imbalance vs n_splits
        if stratified:
            raise ValueError(
                f"StratifiedKFold failed: {e}. Ensure each class has at least 'n_splits' samples or reduce n_splits."
            )
        raise
    return scores.astype(float)


def _r2_score_manual(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Manual R^2: 1 - SS_res / SS_tot.

    - SS_res = sum((y_true - y_pred)^2)
    - SS_tot = sum((y_true - mean(y_true))^2)
    If SS_tot == 0 (constant y), return 0.0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)

# Scorer object for use in cross_val_score
_R2_SCORER = make_scorer(_r2_score_manual, greater_is_better=True)


def linear_regression_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with Ordinary Least Squares regression (R^2 scoring)."""
    params = _STRATEGY_PARAMS.get("linear_regression", {})
    model = LinearRegression(**params)
    return _run_cv(model, X_subset, y, n_splits=n_splits, seed=seed, scoring=_R2_SCORER, stratified=False)


def linear_regression_mae_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Ordinary Least Squares regression scored by negative MAE (higher is better)."""
    params = _STRATEGY_PARAMS.get("linear_regression", {})
    model = LinearRegression(**params)
    return _run_cv(model, X_subset, y, n_splits=n_splits, seed=seed, scoring="neg_mean_absolute_error", stratified=False)


def linear_regression_mse_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Ordinary Least Squares regression scored by negative MSE (higher is better)."""
    params = _STRATEGY_PARAMS.get("linear_regression", {})
    model = LinearRegression(**params)
    return _run_cv(model, X_subset, y, n_splits=n_splits, seed=seed, scoring="neg_mean_squared_error", stratified=False)


def random_forest_regression_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with RandomForestRegressor using R^2 scoring."""
    rf_params = {"n_estimators": 200, "random_state": seed, "n_jobs": 1}
    rf_params.update(_STRATEGY_PARAMS.get("random_forest_regression", {}))
    model = RandomForestRegressor(**cast(Dict[str, Any], rf_params))
    return _run_cv(model, X_subset, y, n_splits=n_splits, seed=seed, scoring=_R2_SCORER, stratified=False)


def gradient_boosting_regression_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with GradientBoostingRegressor using R^2 scoring."""
    params = {"random_state": seed}
    params.update(_STRATEGY_PARAMS.get("gradient_boosting_regression", {}))
    model = GradientBoostingRegressor(**cast(Dict[str, Any], params))
    return _run_cv(model, X_subset, y, n_splits=n_splits, seed=seed, scoring=_R2_SCORER, stratified=False)


def svr_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with Support Vector Regressor using R^2 scoring."""
    params = _STRATEGY_PARAMS.get("svr", {})
    model = SVR(**cast(Dict[str, Any], params))
    return _run_cv(model, X_subset, y, n_splits=n_splits, seed=seed, scoring=_R2_SCORER, stratified=False)


def logistic_regression_accuracy_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with LogisticRegression using accuracy scoring."""
    if np.unique(y).size < 2:
        raise ValueError("Logistic regression fitness requires at least two classes in y.")

    params = {"max_iter": 1000, "solver": "lbfgs"}
    params.update(_STRATEGY_PARAMS.get("logistic_regression", {}))
    model = LogisticRegression(**cast(Dict[str, Any], params))
    return _run_cv(model, X_subset, y, n_splits=n_splits, seed=seed, scoring="accuracy", stratified=True)


def logistic_regression_precision_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with LogisticRegression using macro-averaged precision."""
    if np.unique(y).size < 2:
        raise ValueError("Logistic regression precision fitness requires at least two classes in y.")

    params = {"max_iter": 1000, "solver": "lbfgs"}
    params.update(_STRATEGY_PARAMS.get("logistic_regression", {}))
    model = LogisticRegression(**cast(Dict[str, Any], params))
    return _run_cv(
        model,
        X_subset,
        y,
        n_splits=n_splits,
        seed=seed,
        scoring="precision_macro",
        stratified=True,
    )


def logistic_regression_log_loss_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with LogisticRegression using (negative) log loss."""
    if np.unique(y).size < 2:
        raise ValueError("Logistic regression log-loss fitness requires at least two classes in y.")

    params = {"max_iter": 1000, "solver": "lbfgs"}
    params.update(_STRATEGY_PARAMS.get("logistic_regression", {}))
    model = LogisticRegression(**cast(Dict[str, Any], params))
    return _run_cv(
        model,
        X_subset,
        y,
        n_splits=n_splits,
        seed=seed,
        scoring="neg_log_loss",
        stratified=True,
    )


def logistic_regression_recall_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with LogisticRegression using macro-averaged recall."""
    if np.unique(y).size < 2:
        raise ValueError("Logistic regression recall fitness requires at least two classes in y.")

    params = {"max_iter": 1000, "solver": "lbfgs"}
    params.update(_STRATEGY_PARAMS.get("logistic_regression", {}))
    model = LogisticRegression(**cast(Dict[str, Any], params))
    return _run_cv(
        model,
        X_subset,
        y,
        n_splits=n_splits,
        seed=seed,
        scoring="recall_macro",
        stratified=True,
    )


def logistic_regression_roc_auc_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with LogisticRegression using ROC-AUC scoring (binary/multiclass-handled by sklearn)."""
    if np.unique(y).size < 2:
        raise ValueError("Logistic regression ROC-AUC fitness requires at least two classes in y.")
    params = {"max_iter": 1000, "solver": "lbfgs"}
    params.update(_STRATEGY_PARAMS.get("logistic_regression", {}))
    model = LogisticRegression(**cast(Dict[str, Any], params))
    return _run_cv(model, X_subset, y, n_splits=n_splits, seed=seed, scoring="roc_auc", stratified=True)


def _random_forest_classification_cv(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
    scoring: str,
) -> FitnessScores:
    if np.unique(y).size < 2:
        raise ValueError("Random forest classification fitness requires at least two classes in y.")

    rf_params = {"n_estimators": 300, "max_depth": None, "random_state": seed, "n_jobs": 1}
    rf_params.update(_STRATEGY_PARAMS.get("random_forest_classification", {}))
    model = RandomForestClassifier(**cast(Dict[str, Any], rf_params))
    return _run_cv(model, X_subset, y, n_splits=n_splits, seed=seed, scoring=scoring, stratified=True)


def random_forest_classification_accuracy_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with RandomForestClassifier using accuracy scoring."""
    return _random_forest_classification_cv(
        X_subset,
        y,
        n_splits,
        seed,
        scoring="accuracy",
    )


def random_forest_classification_precision_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with RandomForestClassifier using macro-averaged precision."""
    return _random_forest_classification_cv(
        X_subset,
        y,
        n_splits,
        seed,
        scoring="precision_macro",
    )


def random_forest_classification_recall_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with RandomForestClassifier using macro-averaged recall."""
    return _random_forest_classification_cv(
        X_subset,
        y,
        n_splits,
        seed,
        scoring="recall_macro",
    )


def random_forest_classification_log_loss_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with RandomForestClassifier using (negative) log loss."""
    return _random_forest_classification_cv(
        X_subset,
        y,
        n_splits,
        seed,
        scoring="neg_log_loss",
    )


# Backwards compatibility alias (legacy name without explicit metric)
random_forest_classification_fitness = random_forest_classification_accuracy_fitness


def gradient_boosting_classification_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with GradientBoostingClassifier using accuracy scoring."""
    if np.unique(y).size < 2:
        raise ValueError("Gradient boosting classification fitness requires at least two classes in y.")

    params = {"random_state": seed}
    params.update(_STRATEGY_PARAMS.get("gradient_boosting_classification", {}))
    model = GradientBoostingClassifier(**cast(Dict[str, Any], params))
    return _run_cv(model, X_subset, y, n_splits=n_splits, seed=seed, scoring="accuracy", stratified=True)


def svc_fitness(
    X_subset: FloatArray,
    y: FloatArray,
    n_splits: int,
    seed: int,
) -> FitnessScores:
    """Evaluate subsets with Support Vector Classifier using accuracy scoring."""
    if np.unique(y).size < 2:
        raise ValueError("SVC fitness requires at least two classes in y.")

    # SVC has no random_state parameter; leave RNG to data shuffling
    params = _STRATEGY_PARAMS.get("svc", {})
    model = SVC(**cast(Dict[str, Any], params))
    return _run_cv(model, X_subset, y, n_splits=n_splits, seed=seed, scoring="accuracy", stratified=True)


register_fitness_strategy("linear_regression", linear_regression_fitness)
register_fitness_strategy("ols", linear_regression_fitness)
register_fitness_strategy("linear_regression_mae", linear_regression_mae_fitness)
register_fitness_strategy("linear_regression_mse", linear_regression_mse_fitness)
register_fitness_strategy("random_forest_regression", random_forest_regression_fitness)
register_fitness_strategy("rf_regression", random_forest_regression_fitness)
register_fitness_strategy("gradient_boosting_regression", gradient_boosting_regression_fitness)
register_fitness_strategy("svr", svr_fitness)

register_fitness_strategy("logistic_regression", logistic_regression_accuracy_fitness)
register_fitness_strategy("logistic_regression_accuracy", logistic_regression_accuracy_fitness)
register_fitness_strategy("logistic_regression_roc_auc", logistic_regression_roc_auc_fitness)
register_fitness_strategy("random_forest_classification", random_forest_classification_accuracy_fitness)
register_fitness_strategy("random_forest_classification_accuracy", random_forest_classification_accuracy_fitness)
register_fitness_strategy("rf_classification", random_forest_classification_accuracy_fitness)
register_fitness_strategy("gradient_boosting_classification", gradient_boosting_classification_fitness)
register_fitness_strategy("svc", svc_fitness)
register_fitness_strategy("logistic_regression_precision", logistic_regression_precision_fitness)
register_fitness_strategy("logistic_regression_log_loss", logistic_regression_log_loss_fitness)
register_fitness_strategy("logistic_regression_recall", logistic_regression_recall_fitness)
register_fitness_strategy("random_forest_classification_precision", random_forest_classification_precision_fitness)
register_fitness_strategy("random_forest_classification_recall", random_forest_classification_recall_fitness)
register_fitness_strategy("random_forest_classification_log_loss", random_forest_classification_log_loss_fitness)


__all__ = [
    "gradient_boosting_classification_fitness",
    "gradient_boosting_regression_fitness",
    "linear_regression_fitness",
    "linear_regression_mae_fitness",
    "linear_regression_mse_fitness",
    "logistic_regression_accuracy_fitness",
    "logistic_regression_log_loss_fitness",
    "logistic_regression_precision_fitness",
    "logistic_regression_recall_fitness",
    "logistic_regression_roc_auc_fitness",
    "random_forest_classification_accuracy_fitness",
    "random_forest_classification_log_loss_fitness",
    "random_forest_classification_precision_fitness",
    "random_forest_classification_recall_fitness",
    "random_forest_classification_fitness",
    "random_forest_regression_fitness",
    "svc_fitness",
    "svr_fitness",
    "set_strategy_params",
    "enable_fixed_cv_splits",
    "disable_fixed_cv_splits",
]
