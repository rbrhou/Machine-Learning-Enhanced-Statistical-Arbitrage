import numpy as np
import pandas as pd


def extract_idiosyncratic_residuals(
    returns: pd.DataFrame, factor_returns: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """Regresses asset returns against systematic PCA factors to extract residuals
    and compute cumulative synthetic spreads.

    :param returns: Standardized asset returns (T x N)
    :param factor_returns: Top systematic factor returns (T x K)
    :return: (daily_residuals, cumulative_spreads)
    """
    
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns)

    # Add constant for OLS intercept (alpha)
    F = factor_returns.to_numpy(dtype=float)
    F_design = np.column_stack([np.ones(F.shape[0]), F])

    # Run multi-factor regression for each individual asset
    for col in returns.columns:
        y = returns[col].to_numpy(dtype=float)
        # OLS estimation: beta = (F^T F)^(-1) F^T y
        betas, _, _, _ = np.linalg.lstsq(F_design, y, rcond=None)
        # Residuals: epsilon_t = y_t - (alpha + sum(beta_k * F_k,t))
        predicted = np.dot(F_design, betas)
        residuals[col] = y - predicted

    # Cumulatively sum residuals over time to construct continuous spread series
    cumulative_spreads = residuals.cumsum()

    return residuals.astype(float), cumulative_spreads.astype(float)
