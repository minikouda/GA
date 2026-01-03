import numpy as np
import pytest

from GA.GA import GeneticSelector
from GA.fitness import get_fitness_strategy, list_available_strategies


def _make_regression_data(n_samples: int = 40, n_features: int = 6, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    true_coef = np.array([1.5, -2.0, 0.0, 0.7, 0.0, 1.2])[:n_features]
    noise = 0.2 * rng.randn(n_samples)
    y = X @ true_coef + noise
    return X, y


def _make_classification_data(n_samples: int = 60, n_features: int = 8, seed: int = 1):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    weights = np.array([2.0, -1.0, 0.0, 1.5, 0.0, 0.0, -0.5, 0.8])[:n_features]
    logits = X @ weights + 0.5 * rng.randn(n_samples)
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    if y.min() == y.max():  # ensure at least two classes
        y[0] = 1 - y[0]
    return X, y


def test_strategy_registry_contains_defaults():
    strategies = list_available_strategies()
    assert "linear_regression" in strategies
    assert "logistic_regression" in strategies
    assert "random_forest_regression" in strategies
    assert "gradient_boosting_regression" in strategies
    assert "gradient_boosting_classification" in strategies
    assert "svr" in strategies
    assert "svc" in strategies


def test_linear_strategy_scores_shape():
    strategy = get_fitness_strategy("linear_regression")
    X, y = _make_regression_data(seed=5)
    scores = strategy(X, y, 3, 42)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    assert np.mean(scores) > 0.6  # signal present => decent R^2


def test_linear_mae_mse_strategies_exist_and_score_shape():
    X, y = _make_regression_data(seed=11)
    mae = get_fitness_strategy("linear_regression_mae")(X, y, 3, 101)
    mse = get_fitness_strategy("linear_regression_mse")(X, y, 3, 101)
    assert mae.shape == (3,) and mse.shape == (3,)
    assert np.isfinite(mae).all() and np.isfinite(mse).all()


def test_linear_mae_matches_sklearn_per_fold():
    """MAE strategy should match sklearn's neg_mean_absolute_error per fold."""
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.linear_model import LinearRegression

    X, y = _make_regression_data(seed=12)
    n_splits = 4
    seed = 123
    strategy = get_fitness_strategy("linear_regression_mae")
    ga_scores = strategy(X, y, n_splits, seed)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sk_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="neg_mean_absolute_error")

    assert ga_scores.shape == sk_scores.shape
    assert np.allclose(ga_scores, sk_scores, atol=1e-12)


def test_linear_mse_matches_sklearn_per_fold():
    """MSE strategy should match sklearn's neg_mean_squared_error per fold."""
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.linear_model import LinearRegression

    X, y = _make_regression_data(seed=13)
    n_splits = 5
    seed = 321
    strategy = get_fitness_strategy("linear_regression_mse")
    ga_scores = strategy(X, y, n_splits, seed)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sk_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="neg_mean_squared_error")

    assert ga_scores.shape == sk_scores.shape
    assert np.allclose(ga_scores, sk_scores, atol=1e-12)


def test_ga_can_use_random_forest_strategy():
    X, y = _make_regression_data(seed=2)
    ga = GeneticSelector(
        X,
        y,
        pop_size=10,
        n_gen=10,
        n_splits=2,
        mutation_rate=0.05,
        fitness_strategy="random_forest_regression",
        random_state=5,
    )
    best_ind, best_score, best_fit = ga.run()
    print(best_score)
    assert best_ind.shape == (X.shape[1],)
    assert isinstance(best_score, float)
    assert isinstance(best_fit, float)
    assert best_score > 0.5  # decent R^2 on structured data


def test_ga_can_use_logistic_strategy():
    X, y = _make_classification_data(seed=3)
    ga = GeneticSelector(
        X,
        y,
        pop_size=10,
        n_gen=3,
        n_splits=2,
        mutation_rate=0.05,
        fitness_strategy="logistic_regression",
        random_state=5,
    )
    best_ind, best_score, best_fit = ga.run()
    assert best_ind.shape == (X.shape[1],)
    assert isinstance(best_score, float)
    assert isinstance(best_fit, float)
    assert best_score > 0.5  # better than random guessing on structured data


def test_logistic_regression_roc_auc_strategy():
    X, y = _make_classification_data(seed=10)
    strategy = get_fitness_strategy("logistic_regression_roc_auc")
    scores = strategy(X, y, 3, 12)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    assert np.mean(scores) > 0.5


def test_logistic_regression_roc_auc_matches_sklearn_per_fold():
    """ROC-AUC strategy should match sklearn's roc_auc per fold with identical splits."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression

    X, y = _make_classification_data(seed=14)
    n_splits = 3
    seed = 777
    strategy = get_fitness_strategy("logistic_regression_roc_auc")
    ga_scores = strategy(X, y, n_splits, seed)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sk_scores = cross_val_score(LogisticRegression(max_iter=1000, solver="lbfgs"), X, y, cv=cv, scoring="roc_auc")

    assert ga_scores.shape == sk_scores.shape
    assert np.allclose(ga_scores, sk_scores, atol=1e-12)


def test_ga_can_use_gradient_boosting_regression_strategy():
    X, y = _make_regression_data(seed=4)
    ga = GeneticSelector(
        X,
        y,
        pop_size=10,
        n_gen=5,
        n_splits=2,
        mutation_rate=0.05,
        fitness_strategy="gradient_boosting_regression",
        random_state=6,
    )
    best_ind, best_score, best_fit = ga.run()
    assert best_ind.shape == (X.shape[1],)
    assert isinstance(best_score, float)
    assert isinstance(best_fit, float)
    assert best_score > 0.1  # Gradient boosting can be sensitive


@pytest.mark.slow
def test_ga_can_use_svr_strategy():
    X, y = _make_regression_data(seed=5)
    ga = GeneticSelector(
        X,
        y,
        pop_size=10,
        n_gen=5,
        n_splits=2,
        mutation_rate=0.05,
        fitness_strategy="svr",
        random_state=7,
    )
    best_ind, best_score, best_fit = ga.run()
    assert best_ind.shape == (X.shape[1],)
    assert isinstance(best_score, float)
    assert isinstance(best_fit, float)
    assert best_score > 0.1  # SVR can be sensitive


def test_ga_can_use_gradient_boosting_classification_strategy():
    X, y = _make_classification_data(seed=6)
    ga = GeneticSelector(
        X,
        y,
        pop_size=10,
        n_gen=3,
        n_splits=2,
        mutation_rate=0.05,
        fitness_strategy="gradient_boosting_classification",
        random_state=8,
    )
    best_ind, best_score, best_fit = ga.run()
    assert best_ind.shape == (X.shape[1],)
    assert isinstance(best_score, float)
    assert isinstance(best_fit, float)
    assert best_score > 0.5


@pytest.mark.slow
def test_ga_can_use_svc_strategy():
    X, y = _make_classification_data(seed=7)
    ga = GeneticSelector(
        X,
        y,
        pop_size=10,
        n_gen=3,
        n_splits=2,
        mutation_rate=0.05,
        fitness_strategy="svc",
        random_state=9,
    )
    best_ind, best_score, best_fit = ga.run()
    assert best_ind.shape == (X.shape[1],)
    assert isinstance(best_score, float)
    assert isinstance(best_fit, float)
    assert best_score > 0.5