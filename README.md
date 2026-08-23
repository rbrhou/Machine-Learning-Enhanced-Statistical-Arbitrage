# Machine Learning-Enhanced PCA Statistical Arbitrage

An institutional-grade quantitative trading engine implementing the statistical arbitrage framework formalized by Avellaneda & Lee (2010), augmented with unsupervised machine learning (DBSCAN) and continuous-time stochastic calibration.

---

## 1. Overview & Theoretical Framework

Traditional statistical arbitrage often relies on static 1-to-1 pairs trading. This project implements a multi-asset systematic framework that trades an individual equity against a dynamic basket of systematic eigenfactors:

* **Dimensionality Reduction:** Compresses thousands of correlated returns into a small set of orthogonal systematic risk drivers using Principal Component Analysis (PCA).
* **Dynamic Asset Clustering:** Applies DBSCAN clustering directly to PCA factor loadings to identify cohesive asset cohorts sharing identical factor sensitivities without relying on fundamental sector classifications.
* **Systematic Risk Hedging:** Isolates purely idiosyncratic, stock-specific residual spreads by regressing asset returns against the top PCA factor returns.
* **Statistical Diagnostics:** Enforces stationarity via Augmented Dickey-Fuller (ADF) unit-root tests and screens for negative 1st-order autocorrelation to verify mean-reverting behavior before deploying capital[cite: 1, 5].
* **Continuous-Time SDE Calibration:** Models idiosyncratic residual spreads using the Ornstein-Uhlenbeck (OU) process, discretized via an $\text{AR}(1)$ framework to estimate mean-reversion speeds, equilibrium targets, and dimensionless $s$-score trade signals.

---

## 2. Mathematical Architecture

### Phase 1: Covariance Mapping & Eigendecomposition
For a standardized, mean-centered return matrix $X \in \mathbb{R}^{T \times N}$, the sample covariance matrix $\Sigma$ is computed as:
$$\Sigma = \frac{1}{T - 1} X^T X$$


Solving the eigenvalue problem extracts systematic factor loadings ($v_i$) and eigenvalues ($\lambda_i$):
$$\Sigma v_i = \lambda_i v_i$$


The top $K$ eigenvectors form synthetic systematic factor return series:
$$F = X V_K$$

### Phase 2: Idiosyncratic Residual Isolation
Individual asset returns $R_i$ are regressed against systematic factor returns $F$ via Ordinary Least Squares (OLS):
$$R_{i,t} = \alpha_i + \sum_{k=1}^K \beta_{i,k} F_{k,t} + \epsilon_{i,t}$$


The daily residual $\epsilon_{i,t}$ is cumulatively integrated to form the continuous synthetic spread $x_t$:
$$x_t = \sum_{s=1}^t \epsilon_{i,s}$$


### Phase 3: Continuous-Time Ornstein-Uhlenbeck Calibration
The continuous spread $x_t$ is modeled as a mean-reverting stochastic differential equation:
$$dx_t = \kappa(\theta - x_t)dt + \sigma dW_t$$


* $\kappa$: Mean-reversion speed ($\tau_{1/2} = \frac{\ln(2)}{\kappa}$)
* $\theta$: Long-term equilibrium spread
* $\sigma$: Diffusion volatility of random market noise

Discretized as an $\text{AR}(1)$ specification over time step $\Delta t = \frac{1}{252}$:
$$x_n = a + b x_{n-1} + \zeta_n, \quad \zeta_n \sim \mathcal{N}(0, \sigma_\zeta^2)$$


Continuous parameters are recovered via exact discrete-to-continuous transformations:
$$\kappa = -\frac{\ln(b)}{\Delta t}, \quad \theta = \frac{a}{1 - b}, \quad \sigma_{\text{eq}} = \frac{\sigma_\zeta}{\sqrt{1 - b^2}}$$

### Phase 4: Signal Generation Logic
Trading signals are generated using the standardized, dimensionless $s$-score:
$$s_t = \frac{x_t - \theta}{\sigma_{\text{eq}}}$$


* **Open Long Spread ($+1$):** $s_t < -s_{\text{open}}$ (Spread is oversold; buy asset, short factor basket).
* **Open Short Spread ($-1$):** $s_t > +s_{\text{open}}$ (Spread is overbought; short asset, buy factor basket).
* **Close Position ($0$):** $\vert{}s_t\vert{} \le s_{\text{close}}$ (Spread has reverted to equilibrium $\theta$).

---

## 3. Project Structure

```text
quant-pca-stat-arb/
├── data/                  # Local parquet/csv data cache (gitignored)
│   └── .gitkeep
├── notebooks/             # Exploratory research & interactive backtests
│   └── 01_factor_decomposition.ipynb
├── src/                   # Production source code
│   ├── __init__.py
│   ├── backtest.py        # Signal generation & vectorized PnL evaluation
│   ├── clustering.py      # Dynamic DBSCAN clustering on PCA loadings
│   ├── data_loader.py     # yfinance data ingestion and parquet caching
│   ├── diagnostics.py     # ADF stationarity & autocorrelation checks
│   ├── ou_process.py      # Continuous OU calibration & s-score computation
│   ├── pca_model.py       # Covariance computation & eigendecomposition
│   └── residuals.py       # OLS residual extraction & cumulative spreads
├── tests/                 # Unit tests for numerical stability
│   └── test_engine.py
├── .gitignore             # Git exclusion rules for large datasets/caches
├── README.md              # Project documentation
└── requirements.txt       # Environment dependencies
