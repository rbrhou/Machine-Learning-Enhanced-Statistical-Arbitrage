import numpy as np
import pandas as pd


class PortfolioBacktester:

    def __init__(
        self,
        s_open: float = 1.25,
        s_close: float = 0.5,
        transaction_cost_bps: float = 5.0,
        max_gross_leverage: float = 1.0,
    ):
        """Initializes the Multi-Asset Statistical Arbitrage Backtester.

        :param s_open: S-score entry threshold.
        :param s_close: S-score exit threshold.
        :param transaction_cost_bps: Proportional cost per turnover in basis
        points (e.g., 5.0 bps = 0.0005).
        :param max_gross_leverage: Maximum sum of absolute portfolio weights
        (|w_i|).
        """
        self.s_open = s_open
        self.s_close = s_close
        self.tc_rate = transaction_cost_bps / 10000.0
        self.max_gross_leverage = max_gross_leverage

    def generate_signals(self, s_scores: pd.DataFrame) -> pd.DataFrame:
        """Generates continuous discrete position signals (-1, 0, 1) across all assets."""
        signals = pd.DataFrame(0, index=s_scores.index, columns=s_scores.columns)

        for col in s_scores.columns:
            series = s_scores[col].values
            pos = np.zeros(len(series), dtype=int)
            curr = 0

            for t in range(len(series)):
                s = series[t]
                if np.isnan(s):
                    pos[t] = 0
                    continue

                if curr == 0:
                    if s < -self.s_open:
                        curr = 1  # Long undervalued spread
                    elif s > self.s_open:
                        curr = -1  # Short overbought spread
                elif curr == 1:
                    if s >= -self.s_close:
                        curr = 0  # Reversion complete
                elif curr == -1:
                    if s <= self.s_close:
                        curr = 0  # Reversion complete

                pos[t] = curr

            signals[col] = pos

        return signals

    def compute_portfolio_weights(
        self, signals: pd.DataFrame, sigma_eq_dict: dict[str, float]
    ) -> pd.DataFrame:
        """Weights positions by inverse equilibrium volatility (1 / sigma_eq)

        and normalizes row-wise to enforce gross leverage limits.
        """
        raw_weights = pd.DataFrame(
            0.0, index=signals.index, columns=signals.columns
        )

        # 1. Scale by 1 / sigma_eq
        for col in signals.columns:
            s_eq = sigma_eq_dict.get(col, np.nan)
            if not np.isnan(s_eq) and s_eq > 0:
                raw_weights[col] = signals[col] / s_eq

        # 2. Row-wise normalization to respect gross leverage
        gross_sum = raw_weights.abs().sum(axis=1)
        gross_sum = gross_sum.replace(0.0, np.nan)

        normalized_weights = raw_weights.div(gross_sum, axis=0) * min(
            1.0, self.max_gross_leverage
        )
        return normalized_weights.fillna(0.0)

    def run_backtest(
        self, weights: pd.DataFrame, daily_residuals: pd.DataFrame
    ) -> pd.DataFrame:
        """Executes portfolio backtest under T+1 causal execution,

        deducting transaction costs from rebalancing turnover.
        """
        # Enforce causality: weights decided at t-1 execute on returns at t
        lagged_weights = weights.shift(1).fillna(0.0)

        # Gross portfolio return = sum(w_{i, t-1} * e_{i, t})
        gross_returns = (lagged_weights * daily_residuals).sum(axis=1)

        # Turnover = sum(|w_{i, t} - w_{i, t-1}|)
        turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
        transaction_costs = turnover * self.tc_rate

        net_returns = gross_returns - transaction_costs

        results = pd.DataFrame(
            {
                "gross_returns": gross_returns,
                "transaction_costs": transaction_costs,
                "net_returns": net_returns,
                "turnover": turnover,
                "gross_equity": (1.0 + gross_returns).cumprod(),
                "net_equity": (1.0 + net_returns).cumprod(),
            },
            index=weights.index,
        )

        return results

    @staticmethod
    def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
        """Calculates institutional risk and performance metrics."""
        clean_ret = returns.dropna()
        n_days = len(clean_ret)
        if n_days < 2:
            return {}

        ann_factor = 252
        cum_ret = (1.0 + clean_ret).cumprod()
        total_return = cum_ret.iloc[-1] - 1.0
        cagr = (cum_ret.iloc[-1]) ** (ann_factor / n_days) - 1.0

        ann_mean = clean_ret.mean() * ann_factor
        ann_vol = clean_ret.std() * np.sqrt(ann_factor)

        excess_ret = clean_ret - (risk_free_rate / ann_factor)
        sharpe = (
            (excess_ret.mean() * ann_factor) / ann_vol if ann_vol > 0 else 0.0
        )

        downside_ret = clean_ret[clean_ret < 0]
        downside_vol = downside_ret.std() * np.sqrt(ann_factor)
        sortino = (
            (excess_ret.mean() * ann_factor) / downside_vol
            if downside_vol > 0
            else 0.0
        )

        # Drawdowns
        running_max = cum_ret.cummax()
        drawdowns = (cum_ret - running_max) / running_max
        max_drawdown = drawdowns.min()
        calmar = cagr / abs(max_drawdown) if abs(max_drawdown) > 0 else 0.0

        return {
            "Total Return": total_return,
            "CAGR": cagr,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": max_drawdown,
            "Calmar Ratio": calmar,
            "Win Rate": (clean_ret > 0).mean(),
        }
