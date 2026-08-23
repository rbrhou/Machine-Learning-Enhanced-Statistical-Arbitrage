import numpy as np
import pandas as pd
from src.ou_process import OUProcessModel
from src.pca_model import PCAFactorModel


def test_pca_dimension_reduction():
    # Synthetic random data (100 days x 5 assets)
    returns = pd.DataFrame(np.random.normal(0, 0.02, (100, 5)))
    pca = PCAFactorModel(n_components=2)
    pca.fit(returns)

    assert pca.factor_loadings.shape == (5, 2)
    assert pca.factor_returns.shape == (100, 2)
    assert np.isclose(pca.explained_variance_ratio.sum(), 1.0)


def test_ou_calibration_mean_reverting():
    # Generate synthetic stationary AR(1) process: x_t = 0.8 * x_{t-1} + noise
    np.random.seed(42)
    T = 500
    spread = np.zeros(T)
    for t in range(1, T):
        spread[t] = 0.8 * spread[t - 1] + np.random.normal(0, 0.1)

    spread_series = pd.Series(spread)
    ou_model = OUProcessModel(dt=1.0 / 252.0)
    params = ou_model.fit_spread(spread_series)

    assert not np.isnan(params["kappa"])
    assert params["kappa"] > 0
    assert params["half_life"] > 0
