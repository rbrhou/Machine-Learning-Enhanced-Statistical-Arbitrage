import numpy as np
import pandas as pd


def generate_trading_signals(
    s_scores: pd.Series, s_open: float = 1.25, s_close: float = 0.5
) -> pd.Series:
    
    """Generates discrete positions (-1 for short, 1 for long, 0 for flat)."""
    
    positions = pd.Series(0, index=s_scores.index, dtype=int)
    current_pos = 0

    for i, s in enumerate(s_scores):
        if np.isnan(s):
            positions.iloc[i] = 0
            continue

        if current_pos == 0:
            if s < -s_open:
                current_pos = 1  # Buy undervalued spread
            elif s > s_open:
                current_pos = -1  # Short overextended spread
        elif current_pos == 1:
            if s >= -s_close:
                current_pos = 0  # Close long
        elif current_pos == -1:
            if s <= s_close:
                current_pos = 0  # Close short

        positions.iloc[i] = current_pos

    return positions


def evaluate_strategy_pnl(
    positions: pd.Series, daily_residuals: pd.Series
) -> pd.DataFrame:
    
    """Calculates strategy returns, cumulative equity curve, and Sharpe ratio."""
    
    # Shift positions by 1 day to prevent lookahead bias
    lagged_pos = positions.shift(1).fillna(0)
    strategy_returns = lagged_pos * daily_residuals

    cumulative_pnl = (1.0 + strategy_returns).cumprod()

    # Annualized Performance Metrics
    ann_return = strategy_returns.mean() * 252
    ann_vol = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = ann_return / ann_vol if ann_vol != 0 else 0.0

    return pd.DataFrame(
        {
            "strategy_returns": strategy_returns,
            "equity_curve": cumulative_pnl,
            "sharpe_ratio": sharpe_ratio,
        }
    )


def apply_var_risk_overlay(
    positions: pd.DataFrame,
    var_forecasts: np.ndarray,
    target_risk_limit: float = 0.02,
    var_index_offset: int = 30,
) -> pd.DataFrame:
    
    """Scales portfolio positions inversely to forecasted 5% VaR and halts

    trading if 1% tail risk exceeds maximum allowable risk limits.

    :param positions: Raw signal DataFrame (T x N)
    :param var_forecasts: TCN output array (T - seq_len, 2) where col 0 = 1%
    VaR, col 1 = 5% VaR
    :param target_risk_limit: Maximum permissible 1% daily drawdown threshold
    :param var_index_offset: Sequence length burn-in offset
    """
    
    adjusted_positions = positions.copy()
    valid_dates = positions.index[var_index_offset:]

    for i, date in enumerate(valid_dates):
        var_1pct = abs(var_forecasts[i, 0])
        var_5pct = abs(var_forecasts[i, 1])

        # 1. Circuit Breaker / Risk Halt: Flatten if tail risk breaches budget
        if var_1pct > target_risk_limit:
            adjusted_positions.loc[date] = 0.0
            continue

        # 2. Dynamic Volatility Scaling: Scale inversely by 5% VaR
        scale_factor = (
            min(1.5, target_risk_limit / var_5pct) if var_5pct > 1e-4 else 1.0
        )
        adjusted_positions.loc[date] = positions.loc[date] * scale_factor

    return adjusted_positions
