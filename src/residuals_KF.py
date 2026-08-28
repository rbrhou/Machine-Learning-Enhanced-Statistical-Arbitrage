import numpy as np
import pandas as pd


class KalmanResidualFilter:

    def __init__(
        self,
        n_factors: int,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-3,):
        
        """Initializes a Multi-Factor Kalman Filter for dynamic hedge ratio estimation.

        :param n_factors: Number of systematic PCA factors (K).
        :param process_noise: Process noise variance (sigma_w^2) governing beta
        drift.
        :param measurement_noise: Measurement noise variance (R = sigma_v^2).
        """
        self.dim = n_factors + 1  # Intercept (alpha) + K factor betas
        self.Q = np.eye(self.dim) * process_noise
        self.R = measurement_noise

    def filter_series(
        self, y: np.ndarray, F: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        
        """Runs the Kalman Filter recursion over the historical time series for a
        single asset.

        :param y: Asset return vector (T,)
        :param F: PCA factor returns matrix (T, K)
        :return: (innovations/residuals of shape (T,), dynamic_betas of shape (T, K+1))
        """
        
        T = len(y)
        # Initialize state estimate and covariance matrix
        beta_hat = np.zeros(self.dim)
        P = np.eye(self.dim) * 1.0

        innovations = np.zeros(T)
        betas_history = np.zeros((T, self.dim))

        for t in range(T):
            # Design vector at time t: [1, F_{1,t}, ..., F_{K,t}]
            H_t = np.insert(F[t], 0, 1.0)  # Shape (K+1,)

            # 1. Predict step
            beta_pred = beta_hat
            P_pred = P + self.Q

            # 2. Innovation (Residual) step
            y_pred = np.dot(H_t, beta_pred)
            e_t = y[t] - y_pred  # Idiosyncratic residual
            innovations[t] = e_t

            # Innovation variance: S_t = H_t @ P_pred @ H_t.T + R
            S_t = np.dot(H_t, np.dot(P_pred, H_t)) + self.R

            # 3. Update step (Kalman Gain)
            K_gain = np.dot(P_pred, H_t) / S_t
            beta_hat = beta_pred + K_gain * e_t
            P = P_pred - np.outer(K_gain, H_t) @ P_pred

            betas_history[t] = beta_hat

        return innovations, betas_history


def extract_idiosyncratic_residuals(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    process_noise: float = 1e-4,
    measurement_noise: float = 1e-3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    
    """Filters each asset against systematic PCA factors using Kalman Filters
    to extract time-varying idiosyncratic residuals and cumulative spreads.

    :param returns: Standardized asset returns (T x N)
    :param factor_returns: Systematic factor returns from PCA (T x K)
    :param process_noise: Process noise variance for beta drift
    :param measurement_noise: Measurement noise variance for observation noise
    :return: (daily_residuals, cumulative_spreads, dynamic_betas_dict)
    """
    
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns)
    dynamic_betas_dict = {}

    F = factor_returns.values
    K = factor_returns.shape[1]
    feature_names = ["alpha"] + list(factor_returns.columns)

    for col in returns.columns:
        y = returns[col].values
        kf = KalmanResidualFilter(
            n_factors=K,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )
        innovations, betas_history = kf.filter_series(y, F)

        residuals[col] = innovations
        dynamic_betas_dict[col] = pd.DataFrame(
            betas_history, index=returns.index, columns=feature_names
        )

    # Cumulatively sum the adaptive daily residuals to form continuous spread series
    cumulative_spreads = residuals.cumsum()

    return (
        residuals.astype(float),
        cumulative_spreads.astype(float),
        dynamic_betas_dict,
    )
