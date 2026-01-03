"""Fitness strategy registry and built-in implementations."""
from __future__ import annotations

from .registry import (
    FitnessScores,
    FitnessStrategy,
    get_fitness_strategy,
    list_available_strategies,
    register_fitness_strategy,
)
from .strategies import (
    linear_regression_fitness,
    linear_regression_mae_fitness,
    linear_regression_mse_fitness,
    logistic_regression_accuracy_fitness,
    logistic_regression_log_loss_fitness,
    logistic_regression_precision_fitness,
    logistic_regression_recall_fitness,
    logistic_regression_roc_auc_fitness,
    random_forest_classification_accuracy_fitness,
    random_forest_classification_fitness,
    random_forest_classification_log_loss_fitness,
    random_forest_classification_precision_fitness,
    random_forest_classification_recall_fitness,
    random_forest_regression_fitness,
    set_global_n_workers,
    set_strategy_params,
    enable_fixed_cv_splits,
    disable_fixed_cv_splits,
)

# Backwards-compatibility alias: older code imports logistic_regression_fitness
logistic_regression_fitness = logistic_regression_accuracy_fitness
random_forest_classification_accuracy = random_forest_classification_accuracy_fitness
random_forest_classification_precision = random_forest_classification_precision_fitness
random_forest_classification_recall = random_forest_classification_recall_fitness
random_forest_classification_log_loss = random_forest_classification_log_loss_fitness

__all__ = [
    "FitnessScores",
    "FitnessStrategy",
    "get_fitness_strategy",
    "list_available_strategies",
    "register_fitness_strategy",
    "linear_regression_fitness",
    "linear_regression_mae_fitness",
    "linear_regression_mse_fitness",
    "logistic_regression_accuracy_fitness",
    "logistic_regression_fitness",
    "logistic_regression_log_loss_fitness",
    "logistic_regression_precision_fitness",
    "logistic_regression_recall_fitness",
    "logistic_regression_roc_auc_fitness",
    "random_forest_classification_accuracy_fitness",
    "random_forest_classification_fitness",
    "random_forest_classification_log_loss_fitness",
    "random_forest_classification_precision_fitness",
    "random_forest_classification_recall_fitness",
    "random_forest_regression_fitness",
    "set_global_n_workers",
    "set_strategy_params",
    "enable_fixed_cv_splits",
    "disable_fixed_cv_splits",
]
