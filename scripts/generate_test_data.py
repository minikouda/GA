import numpy as np
import pandas as pd
import os
import argparse

def generate_regression_data(
    n_samples: int = 100,
    n_features: int = 10,
    n_informative: int = 4,
    noise_std: float = 0.001,
    random_state: int = 42,
    output_filename: str = "synthetic_regression_data.csv",
) -> None:
    """
    Generates a synthetic regression dataset with a specified number of informative and noisy features.

    Args:
        n_samples (int): The number of samples to generate.
        n_features (int): The total number of features (informative + noisy).
        n_informative (int): The number of features that will genuinely influence the target.
        noise_std (float): The standard deviation of the Gaussian noise added to the target.
        random_state (int): Seed for reproducibility.
        output_filename (str): The name of the CSV file to save the data.
    """
    if n_informative > n_features:
        raise ValueError("n_informative cannot be greater than n_features.")

    rng = np.random.RandomState(random_state)

    # Generate informative features
    X_informative = rng.rand(n_samples, n_informative)
    # Generate random coefficients for informative features
    true_coef = rng.uniform(-5, 5, n_informative)

    # Generate noisy features
    X_noisy = rng.rand(n_samples, n_features - n_informative)

    # Combine all features
    X = np.hstack((X_informative, X_noisy))

    # Generate target variable y
    # y = (X_informative @ true_coef) + noise
    # Correct calculation for y based on true_coef and informative features
    y = np.array(np.dot(X_informative, true_coef) + rng.normal(0, noise_std, n_samples) > 0, dtype=bool)

    # Create DataFrame
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df_X = pd.DataFrame(X, columns=feature_names)
    df_y = pd.DataFrame(y, columns=["target"])

    df = pd.concat([df_X, df_y], axis=1)

    # Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"Generated synthetic regression dataset with {n_samples} samples and {n_features} features.")
    print(f"Saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic regression dataset as CSV")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of rows/samples")
    parser.add_argument("--n-features", type=int, default=15, help="Total number of features")
    parser.add_argument("--n-informative", type=int, default=5, help="Number of informative features")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Std dev of Gaussian noise added to target")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="GA/data/synthetic_regression_data.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_regression_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        noise_std=args.noise_std,
        random_state=args.random_state,
        output_filename=args.output,
    )
