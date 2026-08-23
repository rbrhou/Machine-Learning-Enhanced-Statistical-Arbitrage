import numpy as np
import pandas as pd


class PCAFactorModel:

    def __init__(self, n_components: int = 5):
        """Initializes the PCA Factor Model.

        :param n_components: Number of top systematic eigenfactors to retain.
        """
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_variance_ratio = None
        self.factor_loadings = None
        self.factor_returns = None

    def fit(self, returns: pd.DataFrame):
        """Computes sample covariance, performs eigendecomposition,

        and extracts factor loadings and returns.
        """
        # Standardize / Mean-center returns (T x N matrix)
        X = returns - returns.mean()
        T, N = X.shape

        # Compute sample covariance matrix: Sigma = (1 / (T - 1)) * X.T @ X
        cov_matrix = np.dot(X.T, X) / (T - 1)

        # Eigendecomposition: Sigma * v_i = lambda_i * v_i
        # eigh is optimized for symmetric/Hermitian matrices
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        # Variance explained ratio
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues / total_variance

        # Retain top K components
        top_eigenvectors = self.eigenvectors[:, : self.n_components]

        # Factor loadings (weights across assets)
        self.factor_loadings = pd.DataFrame(
            top_eigenvectors,
            index=returns.columns,
            columns=[f"PC_{i+1}" for i in range(self.n_components)],
        )

        # Compute synthetic factor return series: F = X @ V
        self.factor_returns = pd.DataFrame(
            np.dot(X, top_eigenvectors),
            index=returns.index,
            columns=[f"PC_{i+1}" for i in range(self.n_components)],
        )

        return self
