# Machine Learning-Enhanced Statistical Arbitrage Engine

An institutional-grade, multi-asset quantitative trading framework that modernizes classical statistical arbitrage. This engine bridges classical linear algebra with automated trading frameworks, unsupervised machine learning, dynamic state-space modeling, and deep learning-based risk management.

Based on the foundational framework by Avellaneda and Lee (2010), this repository replaces rigid static linear regressions with adaptive Kalman Filters, upgrades spatial clustering with Parametric UMAP, and incorporates a PyTorch-based Temporal Convolutional Network (TCN) to forecast tail risk.

---

## Theoretical Framework & Architecture

The quantitative pipeline operates sequentially across four core mathematical domains:

### 1. Market Modeling & Dimensionality Reduction
*   **PCA Factor Extraction:** Compresses thousands of noisy stock movements into a few core market drivers representing systematic risk factors[cite: 1]. 
*   **Data Standardization:** Take your data matrix $X$ (representing a broad universe of daily stock returns) and mean-center it so that each variable has a mean of zero[cite: 1].
*   **Covariance Mapping:** Calculate the covariance matrix $\Sigma$ to map out exactly how every single asset moves in relation to every other asset[cite: 1]:
    $$\Sigma = \frac{1}{n-1} X^T X$$
*   **Eigendecomposition:** Perform eigendecomposition on your covariance matrix to extract its eigenvalues $\lambda_i$ and eigenvectors $v_i$[cite: 1]:
    $$\Sigma v_i = \lambda_i v_i$$
*   **Dynamic Clustering:** You use this unsupervised learning approach to identify clusters of similar stocks dynamically instead of relying on subjective fundamental analysis[cite: 1]. 

### 2. Dynamic Residual Extraction
*   **Multi-Factor Kalman Filter:** Replaces traditional Ordinary Least Squares (OLS) regression because financial relationships are rarely static, and the true beta between two assets drifts over time[cite: 1].
*   **Unobserved State Equation:** You model the unobserved state equation as $\beta_t = \beta_{t-1} + w_t$ (where $w_t$ is process noise)[cite: 1].
*   **Observation Equation:** You model the observation equation as $y_t = \beta_t x_t + v_t$ (where $v_t$ is measurement noise)[cite: 1].
*   **Pure Idiosyncratic Spreads:** The Kalman Filter continuously updates the hidden state as new observations arrive, isolating a pure, adaptive idiosyncratic residual spread[cite: 1].

### 3. Mean-Reversion & Signal Generation
*   **Extract Idiosyncratic Returns:** Hedge out the systematic PCA factors you just calculated[cite: 1]. The residual returns left over represent the purely idiosyncratic, stock-specific behavior that is isolated from broader market noise[cite: 1].
*   **Model the Spread:** Use the Ornstein-Uhlenbeck (OU) process to model the spreading between pairs[cite: 1]. The OU process is a classic Stochastic Differential Equation (SDE) used heavily in statistical arbitrage for modeling the spreading between pairs of assets[cite: 1].
*   **Continuous-Time Mechanics:** The OU process incorporates a deterministic "drift" that pulls the asset value back to its mean, alongside a stochastic "diffusion" that represents random market noise[cite: 1].
*   **Mathematical Equation:** Mathematically, it is often expressed as[cite: 1]:
    $$dx_t = \kappa(\theta - x_t)dt + \sigma dW_t$$
*   **Parameter Calibration:** In this equation, $\kappa$ dictates the speed of mean reversion, $\theta$ is the long-term equilibrium mean, and $\sigma$ scales the random Brownian motion $dW_t$[cite: 1]. 
*   **Exploiting the Arbitrage:** When the spread $x_t$ stretches far beyond $\theta$, the deterministic drift term $\kappa(\theta - x_t)dt$ overpowers the random noise, signaling a high-probability trade to bet on convergence[cite: 1].
*   **Generate Signals:** Use the OU process, or alternatively a mean-reverting neural network, to generate automated entry and exit signals when these idiosyncratic spreads diverge[cite: 1].

### 4. Deep Learning Risk Overlay (TCN)
*   **Temporal Convolutional Network:** A deep learning architecture specifically engineered to handle sequential time-series data without introducing lookahead bias, making it highly effective for financial risk forecasting[cite: 1].
*   **1D Causal Dilated Convolutions:** Ensures the filter output at time $t$ is strictly derived from inputs at time $t$ and earlier, explicitly preventing future data leakage[cite: 1]. For a 1D sequence $\mathbf{x} \in \mathbb{R}^T$ and a convolutional filter $f: \{0, \dots, k-1\} \to \mathbb{R}$ utilizing a dilation factor $d$, the dilated causal convolution is defined mathematically as[cite: 1]:
    $$y_t = (\mathbf{x} *_d f)(t) = \sum_{i=0}^{k-1} f(i) \cdot \mathbf{x}_{t - d \cdot i}$$
*   **Multi-Quantile Pinball Loss:** The network is optimized using a Pinball (or Quantile) loss function to directly estimate the $q$-th conditional quantile of the return distribution, specifically targeting the 1% and 5% tail risks ($q \in \{0.01, 0.05\}$)[cite: 1]. Given a true market observation $y$ and a model prediction $\hat{y}_q$, the loss is computed as[cite: 1]:
    $$\mathcal{L}_q(y, \hat{y}_q) = \max\left(q(y - \hat{y}_q), (q - 1)(y - \hat{y}_q)\right) = (y - \hat{y}_q)\left(q - \mathbb{I}_{\{y < \hat{y}_q\}}\right)$$
*   **Total Batch Loss:** For a multi-quantile forecasting engine evaluating a batch of $N$ samples across target quantiles $Q = \{0.01, 0.05\}$, the final aggregated loss function is[cite: 1]:
    $$\mathcal{L}_{\text{total}} = \sum_{q \in Q} \frac{1}{N} \sum_{i=1}^N \mathcal{L}_q\left(y_i, \hat{y}_{q, i}\right)$$
*   **Statistical Validation:** Compare a classic econometric model, like EGARCH, against the modern time series deep learning model to evaluate predictive accuracy via standard VaR backtesting metrics, such as the Kupiec POF (Proportion of Failures) test[cite: 1].

---

## Repository Structure

*   `data/`: Local data cache (Gitignored except `.gitkeep`).
*   `notebooks/`: 4-Stage Execution Narrative.
    *   `01_factor_decomposition.ipynb`: PCA, Parametric UMAP, and DBSCAN Clustering.
    *   `02_kf_residuals_verification.ipynb`: Kalman Filter tracking vs OLS + OU Calibration.
    *   `03_portfolio_backtest.ipynb`: Multi-Asset Execution & Transaction Cost Friction.
    *   `04_tcn_var_risk_overlay.ipynb`: PyTorch TCN VaR Forecasting & Kupiec POF Testing.
*   `src/`: Core Modular Engine.
    *   `backtest.py`: Portfolio aggregation, inverse-vol weighting, and tearsheets.
    *   `clustering.py`: Parametric UMAP + DBSCAN density clustering.
    *   `data_loader.py`: Local Parquet caching and `yfinance` ingestion.
    *   `diagnostics.py`: ADF stationarity and autocorrelation screening.
    *   `ou_process.py`: Ornstein-Uhlenbeck SDE modeling and $s$-score generation.
    *   `pca_model.py`: Eigendecomposition and variance mapping.
    *   `residuals.py`: Multi-Factor Kalman Filter State-Space model.
    *   `tcn_var.py`: PyTorch TCN architecture and Pinball Loss.
*   `tests/`: Pytest unit testing suite.
*   `.gitignore`: Excludes `data/`, `.pt` weights, and Jupyter checkpoints.
*   `requirements.txt`: Python dependencies.

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/yourusername/stat-arb-quant-engine.git](https://github.com/yourusername/stat-arb-quant-engine.git)
   cd stat-arb-quant-engine
