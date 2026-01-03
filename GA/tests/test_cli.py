import subprocess
import sys
import os
import pytest
import pandas as pd
import numpy as np
from GA import select

# Path to the script: GA/tests/../../scripts/run_ga.py
SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'run_ga.py'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
SAMPLE_CLASS_PATH = os.path.join(DATA_DIR, 'sample_classification_data.csv')

def test_cli_help():
    """Test that the CLI help command runs successfully."""
    if not os.path.exists(SCRIPT_PATH):
        pytest.skip(f"Script not found at {SCRIPT_PATH}")

    result = subprocess.run(
        [sys.executable, SCRIPT_PATH, "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout

def test_cli_run_with_csv(tmp_path):
    """Test running the CLI with a CSV file."""
    if not os.path.exists(SCRIPT_PATH):
        pytest.skip(f"Script not found at {SCRIPT_PATH}")

    # Create a dummy CSV file
    # 50 samples, 5 features
    n_samples = 50
    n_features = 5
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)
    
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    df['target'] = y
    
    data_file = tmp_path / "test_data.csv"
    df.to_csv(data_file, index=False)

    # Run the script
    result = subprocess.run(
        [
            sys.executable, SCRIPT_PATH,
            "--data-path", str(data_file),
            "--pop-size", "10",
            "--n-gen", "2",
            "--n-splits", "2"
        ],
        capture_output=True,
        text=True
    )
    
    # Check if it ran successfully
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0
    assert "GA Selection Result:" in result.stdout
    assert "Selected Features Indices:" in result.stdout

def test_cli_run_default_diabetes():
    """Test running the CLI with default diabetes dataset (no data path)."""
    if not os.path.exists(SCRIPT_PATH):
        pytest.skip(f"Script not found at {SCRIPT_PATH}")

    result = subprocess.run(
        [
            sys.executable, SCRIPT_PATH,
            "--pop-size", "10",
            "--n-gen", "2",
            "--n-splits", "2"
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0
    assert "Using default 'diabetes' dataset" in result.stdout


def test_cli_classification_metric():
    """Test running the CLI for a classification problem with accuracy metric."""
    if not os.path.exists(SCRIPT_PATH):
        pytest.skip(f"Script not found at {SCRIPT_PATH}")
    if not os.path.exists(SAMPLE_CLASS_PATH):
        pytest.skip("Sample classification dataset not available.")

    result = subprocess.run(
        [
            sys.executable, SCRIPT_PATH,
            "--data-path", SAMPLE_CLASS_PATH,
            "--task-type", "classification",
            "--metric", "accuracy",
            "--pop-size", "8",
            "--n-gen", "2",
            "--n-splits", "2"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0
    assert "Cross-validated Score (Accuracy):" in result.stdout


def test_cli_classification_recall():
    """Test running the CLI for a classification problem with recall metric."""
    if not os.path.exists(SCRIPT_PATH):
        pytest.skip(f"Script not found at {SCRIPT_PATH}")
    if not os.path.exists(SAMPLE_CLASS_PATH):
        pytest.skip("Sample classification dataset not available.")

    result = subprocess.run(
        [
            sys.executable, SCRIPT_PATH,
            "--data-path", SAMPLE_CLASS_PATH,
            "--task-type", "classification",
            "--metric", "recall",
            "--pop-size", "8",
            "--n-gen", "2",
            "--n-splits", "2"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0
    assert "Cross-validated Score (Recall):" in result.stdout


@pytest.mark.parametrize(
    "metric,label",
    [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("cross_entropy", "Cross Entropy"),
    ],
)
def test_cli_classification_metric_variants(metric, label):
    """Ensure all supported classification metrics run on the same dataset."""
    if not os.path.exists(SCRIPT_PATH):
        pytest.skip(f"Script not found at {SCRIPT_PATH}")
    if not os.path.exists(SAMPLE_CLASS_PATH):
        pytest.skip("Sample classification dataset not available.")

    result = subprocess.run(
        [
            sys.executable, SCRIPT_PATH,
            "--data-path", SAMPLE_CLASS_PATH,
            "--task-type", "classification",
            "--metric", metric,
            "--pop-size", "6",
            "--n-gen", "2",
            "--n-splits", "2",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0
    assert f"Cross-validated Score ({label}):" in result.stdout


def test_cli_random_forest_recall():
    """Ensure metric-specific random forest strategy is selected when requested."""
    if not os.path.exists(SCRIPT_PATH):
        pytest.skip(f"Script not found at {SCRIPT_PATH}")
    if not os.path.exists(SAMPLE_CLASS_PATH):
        pytest.skip("Sample classification dataset not available.")

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT_PATH,
            "--data-path",
            SAMPLE_CLASS_PATH,
            "--task-type",
            "classification",
            "--fitness",
            "random_forest",
            "--metric",
            "recall",
            "--pop-size",
            "6",
            "--n-gen",
            "2",
            "--n-splits",
            "2",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0
    assert "Cross-validated Score (Recall):" in result.stdout


def test_cli_linear_regression_mae_numeric(tmp_path):
    """CLI score for linear_regression_mae should match library select() result (rounded)."""
    if not os.path.exists(SCRIPT_PATH):
        pytest.skip(f"Script not found at {SCRIPT_PATH}")

    # Create small deterministic regression dataset
    rng = np.random.RandomState(42)
    X = rng.randn(40, 6)
    coef = np.array([1.2, -0.7, 0.0, 0.5, 0.0, 1.0])
    y = X @ coef + 0.1 * rng.randn(40)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    data_file = tmp_path / "lin_mae.csv"
    df.to_csv(data_file, index=False)

    args = [
        sys.executable,
        SCRIPT_PATH,
        "--data-path",
        str(data_file),
        "--fitness",
        "linear_regression_mae",
        "--pop-size",
        "10",
        "--n-gen",
        "3",
        "--n-splits",
        "3",
        "--n-workers",
        "1",
        "--random-state",
        "123",
        "--patience",
        "0",
    ]
    result = subprocess.run(args, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0

    # Parse the score printed by CLI (rounded to 4 decimals)
    import re
    m = re.search(r"Cross-validated Score \([^)]*\):\s*([-0-9.]+)", result.stdout)
    assert m, f"Could not find score in output: {result.stdout}"
    cli_score = float(m.group(1))

    # Compute expected via direct library call with same settings
    expected = select(
        X,
        y,
        fitness_strategy="linear_regression_mae",
        pop_size=10,
        n_gen=3,
        n_splits=3,
        n_workers=1,
        random_state=123,
        patience=0,
    )["R2"]

    assert np.isclose(cli_score, round(expected, 4))


def test_cli_logistic_regression_roc_auc_numeric(tmp_path):
    """CLI score for logistic_regression_roc_auc should match library select() (rounded)."""
    if not os.path.exists(SCRIPT_PATH):
        pytest.skip(f"Script not found at {SCRIPT_PATH}")

    # Create small deterministic binary classification dataset
    rng = np.random.RandomState(7)
    X = rng.randn(60, 8)
    w = np.array([1.5, -0.8, 0.0, 1.2, 0.0, 0.3, -0.5, 0.7])
    logits = X @ w + 0.4 * rng.randn(60)
    y = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
    if y.min() == y.max():
        y[0] = 1 - y[0]

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    data_file = tmp_path / "logit_auc.csv"
    df.to_csv(data_file, index=False)

    args = [
        sys.executable,
        SCRIPT_PATH,
        "--data-path",
        str(data_file),
        "--fitness",
        "logistic_regression_roc_auc",
        "--task-type",
        "classification",
        "--pop-size",
        "8",
        "--n-gen",
        "3",
        "--n-splits",
        "3",
        "--n-workers",
        "1",
        "--random-state",
        "77",
        "--patience",
        "0",
    ]
    result = subprocess.run(args, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0

    import re
    m = re.search(r"Cross-validated Score \([^)]*\):\s*([-0-9.]+)", result.stdout)
    assert m, f"Could not find score in output: {result.stdout}"
    cli_score = float(m.group(1))

    expected = select(
        X,
        y,
        fitness_strategy="logistic_regression_roc_auc",
        pop_size=8,
        n_gen=3,
        n_splits=3,
        n_workers=1,
        random_state=77,
        patience=0,
    )["R2"]

    assert np.isclose(cli_score, round(expected, 4))
