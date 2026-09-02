# Machine Learning-Enhanced PCA Statistical Arbitrage

An institutional-grade quantitative trading engine implementing the statistical arbitrage framework formalized by Avellaneda & Lee (2010), augmented with unsupervised machine learning (DBSCAN) and continuous-time stochastic calibration.

---

## Theoretical Framework & Architecture

The quantitative pipeline operates sequentially across four core mathematical domains:

### 1. Market Modeling & Dimensionality Reduction
*   **PCA Factor Extraction:** Compresses the multi-asset variance of a highly correlated equities universe into orthogonal systematic risk factors (eigenvectors) via eigendecomposition.
*   **Parametric UMAP & DBSCAN Clustering:** Projects linear PCA factor loadings into a dense, non-linear latent manifold using a neural network encoder (Parametric UMAP)[cite: 1]. DBSCAN dynamically isolates highly cohesive, cointegrated asset clusters based on spatial proximity while filtering out erratic assets as noise[cite: 1].

### 2. Dynamic Residual Extraction
*   **Multi-Factor Kalman Filter:** Replaces traditional Ordinary Least Squares (OLS) regression to prevent stale hedge ratios[cite: 1]. The State-Space Model continuously updates unobserved factor betas ($\beta_t = \beta_{t-1} + w_t$) as new daily observations arrive ($y_t = \beta_t x_t + v_t$)[cite: 1].
*   **Pure Idiosyncratic Spreads:** The innovation error ($v_t$) of the Kalman Filter isolates the pure, adaptive idiosyncratic residual spread of each asset, cleanly stripped of broad market influence[cite: 1].

### 3. Mean-Reversion & Signal Generation
*   **Stationarity Diagnostics:** Automated gatekeeping utilizes the Augmented Dickey-Fuller (ADF) test ($p < 0.05$) and lag-1 autocorrelation checks to mathematically prove spread stationarity before capital deployment[cite: 1].
*   **Continuous-Time SDE Calibration:** Models validated spreads using the Ornstein-Uhlenbeck (OU) process ($dx_t = \kappa(\theta - x_t)dt + \sigma dW_t$) to extract the mean-reversion speed ($\kappa$) and generate standardized $s$-scores for automated execution thresholds[cite: 1].

### 4. Deep Learning Risk Overlay (TCN)
*   **Temporal Convolutional Network:** A PyTorch architecture utilizing 1D causal dilated convolutions processes sequential 3D tensors (combining portfolio PnL, squared variance proxies, and macro factors) to forecast next-day Value at Risk (VaR)[cite: 1].
*   **Multi-Quantile Pinball Loss:** Optimizes directly for the 1% and 5% left-tail risk percentiles[cite: 1].
*   **Statistical Validation:** The network's unconditional coverage is formally evaluated against an EGARCH baseline using the Kupiec Proportion of Failures (POF) Likelihood Ratio test[cite: 1].

---

## 2. Theoretical Framework & Architecture

### 1. Market Modeling & Dimensionality Reduction
*   **PCA Factor Extraction:** Compresses the multi-asset variance of a highly correlated equities universe into orthogonal systematic risk factors (eigenvectors) via eigendecomposition. The data matrix $X$ is standardized, and the sample covariance matrix $\Sigma$ is computed to map asset relationships:
$$\Sigma = \frac{1}{n-1} X^T X$$

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

## Repository Structure

```text
├── data/                                   # Local data cache (Gitignored except .gitkeep)
│   └── .gitkeep
├── notebooks/                              # 4-Stage Execution Narrative
│   ├── 01_factor_decomposition.ipynb       # PCA, Parametric UMAP, and DBSCAN Clustering
│   ├── 02_kf_residuals_verification.ipynb  # Kalman Filter tracking vs OLS + OU Calibration
│   ├── 03_portfolio_backtest.ipynb         # Multi-Asset Execution & Transaction Cost Friction
│   └── 04_tcn_var_risk_overlay.ipynb       # PyTorch TCN VaR Forecasting & Kupiec POF Testing
├── src/                                    # Core Modular Engine
│   ├── backtest.py                         # Portfolio aggregation, inverse-vol weighting, and tearsheets
│   ├── clustering.py                       # Parametric UMAP + DBSCAN density clustering
│   ├── data_loader.py                      # Local Parquet caching and yfinance ingestion
│   ├── diagnostics.py                      # ADF stationarity and autocorrelation screening
│   ├── ou_process.py                       # Ornstein-Uhlenbeck SDE modeling and s-score generation
│   ├── pca_model.py                        # Eigendecomposition and variance mapping
│   ├── residuals.py                        # Multi-Factor Kalman Filter State-Space model
│   └── tcn_var.py                          # PyTorch TCN architecture and Pinball Loss
├── tests/                                  # Pytest unit testing suite
│   └── test_engine.py
├── .gitignore                              # Excludes data/, .pt weights, and Jupyter checkpoints
└── requirements.txt                        # Python dependencies
