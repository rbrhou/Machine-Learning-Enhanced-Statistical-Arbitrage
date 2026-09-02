# Machine Learning-Enhanced PCA Statistical Arbitrage

An institutional-grade, market-neutral quantitative trading pipeline that extracts idiosyncratic equity spreads, models mean-reverting alpha, and manages tail risk using deep learning.

The system modernizes the classic Avellaneda & Lee (2010) framework by replacing static regressions with dynamic Kalman Filters, introducing non-linear manifold clustering via Parametric UMAP and DBSCAN, and applying a causal Temporal Convolutional Network (TCN) to dynamically control tail-risk exposure.

---

## Strategy Pipeline Architecture

   **1. Market Factor Decomposition:** Extracts the dominant systematic risk factors driving a broad universe of equities, separating broad macroeconomic trends from individual stock behavior.

   **2. Non-Linear Dimensionality Reduction & Clustering:** Compresses factor exposures into a dense geometric space using Parametric UMAP, allowing DBSCAN to isolate cohesive asset cohorts and discard uncorrelated noise.

   **3. Adaptive Residual Tracking:** Continuously tracks time-varying asset betas using a recursive Kalman Filter state-space model, extracting clean asset-specific mispricings (innovations) without stale lookback bias.

   **4. Statistical Diagnostic Gatekeeper:** Filters spreads through the Augmented Dickey-Fuller (ADF) test and autocorrelation screening to reject non-stationary random walks before capital is allocated.

   **5. Mean-Reversion Signal Engine:** Models stationary spreads as continuous-time mean-reverting processes, converting spread deviations into normalized scores for automated entry and exit triggers.

   **6. Portfolio Execution & Frictions:** Allocates capital using inverse-volatility risk parity across active clusters, accounting for realistic slippage and transaction costs.

   **7. Deep Learning Risk Overlay:** Ingests rolling sequence windows into a causal dilated TCN to forecast next-day 1% and 5% Value at Risk (VaR), dynamically scaling down leverage or halting trades ahead of volatility spikes.

---

## Theoretical Framework & Architecture

### 1. Market Modeling & Dimensionality Reduction
PCA Factor Extraction: Compresses the multi-asset variance of a highly correlated equities universe into orthogonal systematic risk factors (eigenvectors) via eigendecomposition. The data matrix $X$ is standardized, and the sample covariance matrix $\Sigma$ is computed to map asset relationships:

$$\Sigma = \frac{1}{n-1} X^T X$$

Eigendecomposition: The covariance matrix is decomposed to extract its eigenvalues $\lambda_i$ and eigenvectors $v_i$:

$$\Sigma v_i = \lambda_i v_i$$

Parametric UMAP & DBSCAN Clustering: Projects linear PCA factor loadings into a dense, non-linear latent manifold using a neural network encoder. The UMAP algorithm optimizes the fuzzy set cross-entropy loss to contract cohesive assets into dense topological clusters:

$$\mathcal{L}_{\text{UMAP}} = \sum_{e \in E} \left[ w_h(e) \log\left(\frac{w_h(e)}{w_l(e)}\right) + (1 - w_h(e)) \log\left(\frac{1 - w_h(e)}{1 - w_l(e)}\right) \right]$$

Dynamic Selection: DBSCAN evaluates spatial distances on these UMAP embeddings to isolate highly cohesive, cointegrated asset clusters while filtering out erratic assets as noise.

### 2. Dynamic Residual Extraction
Multi-Factor Kalman Filter: Replaces traditional Ordinary Least Squares (OLS) regression to prevent stale hedge ratios. The State-Space Model continuously updates unobserved factor betas as new daily observations arrive.

Unobserved State Equation: Models the dynamic hedge ratio $\beta_t$ as a random walk, where $w_t$ represents the process noise:

$$\beta_t = \beta_{t-1} + w_t$$

Observation Equation: Models the actual market data, where $y_t$ is the real asset return, $x_t$ represents the PCA factor returns, and $v_t$ is the measurement noise:

$$y_t = \beta_t x_t + v_t$$

Pure Idiosyncratic Spreads: The innovation error ($v_t$) of the Kalman Filter isolates the pure, adaptive idiosyncratic residual spread of each asset, cleanly stripped of broad market influence.

### 3. Mean-Reversion & Signal Generation
Once the systematic PCA factors are hedged out, the remaining idiosyncratic residual spread is modeled using the Ornstein-Uhlenbeck (OU) process. This stochastic differential equation (SDE) is governed by two competing forces: a deterministic "drift" that pulls the asset back to its historical mean, and a stochastic "diffusion" representing random market noise[cite: 1, 3].

The continuous-time SDE is defined as:

$$dx_t = \kappa(\theta - x_t)dt + \sigma dW_t$$

Where:
* **$\theta$ (Long-Term Equilibrium Mean):** The historical gravitational center of the trade. Because we are trading hedged residuals derived from zero-mean Kalman innovations, $\theta$ typically centers around zero.
* **$\kappa$ (Mean Reversion Speed):** The deterministic pull or "rubber band" effect. A high $\kappa$ indicates the spread violently snaps back to $\theta$, while a low $\kappa$ indicates sluggish convergence.
* **$\sigma dW_t$ (Stochastic Diffusion):** The unpredictable market noise that continuously perturbs the spread away from equilibrium.

#### Discrete-Time AR(1) Calibration
Because our market data is sampled at discrete daily intervals ($\Delta t = 1/252$) rather than continuously, the SDE is mathematically mapped to an exact Autoregressive AR(1) process for calibration:
$$x_n = a + b x_{n-1} + \zeta_n$$

By fitting the cumulative daily residuals via Ordinary Least Squares (OLS) to this AR(1) structure, we extract the continuous-time parameters:
* **Mean-Reversion Speed:** $\kappa = -\frac{\ln(b)}{\Delta t}$

* **Equilibrium Mean:** $\theta = \frac{a}{1 - b}$

* **Equilibrium Volatility:** $\sigma_{\text{eq}} = \sqrt{\frac{\text{Var}(\zeta)}{1 - b^2}}$

#### Automated Trade Execution
These calibrated parameters create a rigorous mathematical boundary for execution. We transform the raw spread into a dimensionless $s$-score:

$$s_t = \frac{x_t - \theta}{\sigma_{\text{eq}}}$$

When the stochastic diffusion pushes the $s$-score significantly far from zero, the deterministic drift term $\kappa(\theta - x_t)dt$ mathematically overpowers the random noise. This triggers automated entry signals, betting on high-probability convergence back to the historical mean.

### 4. Deep Learning Risk Overlay (TCN)
Temporal Convolutional Network: A PyTorch architecture processing sequential 3D tensors (combining portfolio PnL, squared variance proxies, and macro factors) to forecast next-day Value at Risk (VaR).

1D Causal Dilated Convolutions: Ensures the filter output at time $t$ is strictly derived from inputs at time $t$ and earlier, explicitly preventing future data leakage. For a 1D sequence $\mathbf{x} \in \mathbb{R}^T$ and a convolutional filter $f$ with dilation factor $d$, the operation expands the receptive field efficiently:

$$y_t = (\mathbf{x} *_d f)(t) = \sum_{i=0}^{k-1} f(i) \cdot \mathbf{x}_{t - d \cdot i}$$

Multi-Quantile Pinball Loss: Optimizes directly for the 1% and 5% left-tail risk percentiles ($q \in \{0.01, 0.05\}$). The loss asymmetrically penalizes overestimation and underestimation to pinpoint the conditional quantile:

$$\mathcal{L}_q(y, \hat{y}_q) = \max\left(q(y - \hat{y}_q), (q - 1)(y - \hat{y}_q)\right) = (y - \hat{y}_q)\left(q - \mathbb{I}_{\{y < \hat{y}_q\}}\right)$$

Total Batch Loss: The aggregated loss across a batch of $N$ samples and target quantiles $Q$ is computed as:

$$\mathcal{L}_{\text{total}} = \sum_{q \in Q} \frac{1}{N} \sum_{i=1}^N \mathcal{L}_q\left(y_i, \hat{y}_{q, i}\right)$$

Statistical Validation: The network's unconditional coverage is formally evaluated against an EGARCH baseline using the Kupiec Proportion of Failures (POF) Likelihood Ratio test. Based on empirical failures $x$ over $N$ observations against target risk level $\alpha$, the statistic follows a $\chi^2(1)$ distribution:


$$\text{LR}_{\text{POF}} = -2 \left[ x \ln(\alpha) + (N - x) \ln(1 - \alpha) - x \ln\left(\frac{x}{N}\right) - (N - x) \ln\left(1 - \frac{x}{N}\right) \right]$$

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
