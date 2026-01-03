import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

from GA.fitness import get_fitness_strategy


def _manual_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot == 0.0:
        # For constant targets, define R^2 as 0.0 (not well-defined)
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def test_manual_r2_matches_sklearn_r2_per_fold():
    rng = np.random.RandomState(10)
    X = rng.randn(80, 6)
    w = rng.randn(6)
    y = X @ w + 0.2 * rng.randn(80)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()

    manual_scores = []
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        manual_scores.append(_manual_r2(y[test_idx], y_pred))

    sk_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")

    assert np.allclose(sk_scores, np.array(manual_scores), rtol=1e-12, atol=1e-12)


def test_ga_linear_regression_strategy_uses_manual_r2():
    rng = np.random.RandomState(11)
    X = rng.randn(60, 5)
    w = rng.randn(5)
    y = X @ w + 0.3 * rng.randn(60)

    cv = KFold(n_splits=4, shuffle=True, random_state=123)
    sk_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")

    strategy = get_fitness_strategy("linear_regression")
    ga_scores = strategy(X, y, n_splits=4, seed=123)

    assert ga_scores.shape == (4,)
    assert np.allclose(ga_scores, sk_scores, rtol=1e-12, atol=1e-12)
