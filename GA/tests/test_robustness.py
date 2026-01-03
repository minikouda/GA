import numpy as np
import pytest

from GA.GA import GeneticSelector, select


def test_constant_y_regression():
    rng = np.random.RandomState(0)
    X = rng.rand(30, 6)
    y = np.ones(30)  # constant target
    ga = GeneticSelector(X, y, pop_size=8, n_gen=3, n_splits=3, random_state=1)
    best_ind, best_score, best_fit = ga.run()
    # R^2 with constant y is undefined in sklearn; our strategy returns fold scores, assert finite
    assert isinstance(best_score, float)
    assert np.isfinite(best_fit)


def test_mismatched_lengths_error():
    X = np.random.rand(20, 5)
    y = np.random.rand(19)
    with pytest.raises(ValueError):
        GeneticSelector(X, y)


def test_all_zero_feature_columns():
    rng = np.random.RandomState(1)
    X = rng.rand(40, 5)
    X[:, 2] = 0.0  # constant column
    y = rng.rand(40)
    ga = GeneticSelector(X, y, pop_size=10, n_gen=4, n_splits=3)
    best_ind, best_score, best_fit = ga.run()
    # Should not crash due to correlation NaNs; fitness should penalize but proceed
    assert best_ind.shape == (5,)


def test_logistic_requires_two_classes():
    rng = np.random.RandomState(2)
    X = rng.rand(50, 6)
    y = np.zeros(50)
    ga = GeneticSelector(
        X, y, pop_size=8, n_gen=3, n_splits=3,
        fitness_strategy="logistic_regression",
    )
    with pytest.raises(ValueError):
        # Calling fitness directly to trigger strategy error when subset non-empty
        # Use an individual selecting one feature to avoid empty-individual shortcut
        ind = np.array([1,0,0,0,0,0])
        ga.fitness(ind)


def test_reproducibility_random_state():
    rng = np.random.RandomState(3)
    X = rng.rand(60, 8)
    y = rng.rand(60)
    ga1 = GeneticSelector(X, y, pop_size=12, n_gen=5, random_state=42)
    ga2 = GeneticSelector(X, y, pop_size=12, n_gen=5, random_state=42)
    r1 = ga1.run()
    r2 = ga2.run()
    # With same random_state, results should be identical (bitmask equality)
    assert np.array_equal(r1[0], r2[0])


def test_select_returns_structure():
    X = np.random.rand(25, 7)
    y = np.random.rand(25)
    res = select(X, y, penalty=0.0, pop_size=10, n_gen=3)
    assert set(res.keys()) == {"selected", "R2", "R2pen"}
    assert isinstance(res["R2"], float)
    assert isinstance(res["R2pen"], float)
