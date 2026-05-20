"""
Bayesian Ridge Regression for Baseline Malaria Prevalence Estimation
-------------------------------------------------------------------

This script implements the QR-decomposed Bayesian ridge regression
model described in Equation 3 of the manuscript.

The workflow:
1. Load preprocessed satellite feature embeddings and malaria survey data
2. Apply intervention-based filtering thresholds
3. Construct QR decomposition of the feature matrix
4. Fit Bayesian ridge regression using NumPyro + NUTS
5. Generate posterior predictions
6. Export posterior coefficients for downstream mapping

Model specification:
    tau       ~ Exponential(1)
    beta_hat  ~ Normal(0, tau)
    sigma     ~ Exponential(1)

The QR decomposition is used to improve numerical stability during
MCMC sampling with high-dimensional feature matrices.
"""

# ============================================================
# Imports
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from numpyro.infer import MCMC, NUTS


# ============================================================
# Environment Configuration
# ============================================================

# Use CPU for reproducibility
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Set number of host devices
numpyro.set_host_device_count(2)


# ============================================================
# Configuration Parameters
# ============================================================

# Number of satellite embedding features
N_FEATURES = 1024

# Intervention thresholds
ITN_THRESHOLD = 0.10
ACT_THRESHOLD = 0.50
IRS_THRESHOLD = 0.10

# Temporal filtering
MAX_YEAR = 2008

# MCMC settings
NUM_SAMPLES = 200
NUM_WARMUP = 200
NUM_CHAINS = 1

# Random seed
RANDOM_SEED = 0

# File paths
FEATURE_FILE = "data/baseline_10km_features.csv"
PREDICTION_FILE = "data/prediction_features.csv"

# Output paths
COEFFICIENT_OUTPUT = "outputs/posterior_coefficients.csv"


# ============================================================
# Utility Functions
# ============================================================

def empirical_logit(y, n):
    """
    Compute empirical logit transformation.

    Parameters
    ----------
    y : array-like
        Malaria prevalence values (0-1)
    n : array-like
        Number examined

    Returns
    -------
    array-like
        Empirical logit transformed prevalence
    """
    numerator = y * n + 0.5
    denominator = n * (1 - y) + 0.5

    return np.log(numerator / denominator)


def expit(x):
    """
    Logistic inverse transformation.
    """
    return np.exp(x) / (1 + np.exp(x))


# ============================================================
# Data Loading
# ============================================================

def load_training_data():
    """
    Load and preprocess malaria survey data.

    Returns
    -------
    dict
        Dictionary containing:
        - X : feature matrix
        - y : empirical logit prevalence
        - P : malaria positive counts
        - N : examined counts
        - coordinates : longitude/latitude
    """

    print("Loading training data...")

    # Load feature table
    data = pd.read_csv(FEATURE_FILE)

    # Remove missing prevalence values
    data = data.dropna(subset=["PfPr"])

    # Compute empirical logit prevalence
    data["PfPr_logit"] = empirical_logit(
        data["PfPr"],
        data["Nexamined"]
    )

    # Apply intervention and temporal filters
    data = data[
        (data["itnavg4"] < ITN_THRESHOLD) &
        (data["act"] < ACT_THRESHOLD) &
        (data["irs"] < IRS_THRESHOLD) &
        (data["yearqtr"] < MAX_YEAR)
    ]

    print(f"Filtered observations: {len(data)}")

    # Extract feature matrix
    feature_columns = [
        f"feature_{i}" for i in range(N_FEATURES)
    ]

    X = data[feature_columns].values

    return {
        "X": X,
        "y": data["PfPr_logit"].values,
        "P": data["Npositive"].values,
        "N": data["Nexamined"].values,
        "coordinates": data[["lon_x", "lat_x"]].values
    }


def load_prediction_data():
    """
    Load feature embeddings for prediction locations.

    Returns
    -------
    ndarray
        Prediction feature matrix
    """

    print("Loading prediction features...")

    prediction_data = pd.read_csv(PREDICTION_FILE)

    feature_columns = [
        f"feature_{i}" for i in range(N_FEATURES)
    ]

    return prediction_data[feature_columns].values


# ============================================================
# Bayesian Ridge Regression Model
# ============================================================

def bayesian_ridge_model(X, Q, R_inv, y=None):
    """
    QR-decomposed Bayesian ridge regression model.

    Parameters
    ----------
    X : ndarray
        Original design matrix
    Q : ndarray
        Orthogonal matrix from QR decomposition
    R_inv : ndarray
        Inverse upper triangular matrix
    y : ndarray
        Observed prevalence values
    """

    # --------------------------------------------------------
    # Priors
    # --------------------------------------------------------

    # Global shrinkage parameter
    tau = numpyro.sample(
        "tau",
        dist.Exponential(1)
    )

    # Intercept
    alpha = numpyro.sample(
        "alpha",
        dist.Normal(0, 1)
    )

    # QR-space coefficients
    beta_hat = numpyro.sample(
        "beta_hat",
        dist.Normal(
            jnp.zeros(X.shape[1]),
            tau
        )
    )

    # Observation noise
    sigma = numpyro.sample(
        "sigma",
        dist.Exponential(1)
    )

    # --------------------------------------------------------
    # Linear Predictor
    # --------------------------------------------------------

    mu = numpyro.deterministic(
        "mu",
        alpha + jnp.dot(Q, beta_hat)
    )

    # Transform coefficients back to original feature space
    beta = numpyro.deterministic(
        "beta",
        R_inv @ beta_hat
    )

    # --------------------------------------------------------
    # Likelihood
    # --------------------------------------------------------

    numpyro.sample(
        "y",
        dist.Normal(mu, sigma),
        obs=y
    )


# ============================================================
# Main Workflow
# ============================================================

def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    training_data = load_training_data()

    X = jnp.array(training_data["X"])
    y = jnp.array(training_data["y"])

    X_pred = jnp.array(load_prediction_data())

    # --------------------------------------------------------
    # QR decomposition
    # --------------------------------------------------------

    print("Performing QR decomposition...")

    Q, R = jnp.linalg.qr(X)
    R_inv = jnp.linalg.inv(R)

    # --------------------------------------------------------
    # MCMC inference
    # --------------------------------------------------------

    print("Running Bayesian inference...")

    nuts_kernel = NUTS(
        bayesian_ridge_model,
        target_accept_prob=0.8,
        max_tree_depth=5
    )

    mcmc = MCMC(
        nuts_kernel,
        num_samples=NUM_SAMPLES,
        num_warmup=NUM_WARMUP,
        num_chains=NUM_CHAINS
    )

    mcmc.run(
        jax.random.PRNGKey(RANDOM_SEED),
        X,
        Q,
        R_inv,
        y
    )

    # --------------------------------------------------------
    # Posterior extraction
    # --------------------------------------------------------

    posterior_samples = mcmc.get_samples()

    alpha_samples = posterior_samples["alpha"]
    beta_samples = posterior_samples["beta"]

    # Posterior means
    alpha_mean = alpha_samples.mean(axis=0)
    beta_mean = beta_samples.mean(axis=0)

    # --------------------------------------------------------
    # Training predictions
    # --------------------------------------------------------

    train_predictions = alpha_mean + X @ beta_mean

    correlation = np.corrcoef(
        train_predictions,
        y
    )[0, 1]

    mae = np.mean(
        np.abs(
            expit(train_predictions) - expit(y)
        )
    )

    print("\nModel Performance")
    print("-----------------")
    print(f"Correlation: {correlation:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    # --------------------------------------------------------
    # Prediction generation
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    prediction_values = alpha_mean + X_pred @ beta_mean

    # --------------------------------------------------------
    # Export posterior coefficients
    # --------------------------------------------------------

    print("\nSaving posterior coefficients...")

    coefficient_matrix = np.hstack([
        alpha_samples[:, np.newaxis],
        beta_samples
    ])

    os.makedirs("outputs", exist_ok=True)

    np.savetxt(
        COEFFICIENT_OUTPUT,
        coefficient_matrix,
        delimiter=","
    )

    print(f"Saved coefficients to: {COEFFICIENT_OUTPUT}")

    # --------------------------------------------------------
    # Diagnostic plot
    # --------------------------------------------------------

    plt.figure(figsize=(6, 6))

    plt.scatter(
        train_predictions,
        y,
        alpha=0.4
    )

    plt.xlabel("Predicted prevalence (logit)")
    plt.ylabel("Observed prevalence (logit)")
    plt.title("Predicted vs Observed Prevalence")

    plt.tight_layout()
    plt.show()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()