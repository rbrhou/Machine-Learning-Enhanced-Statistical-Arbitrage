import numpy as np
import pandas as pd

from src.pca_model import PCAFactorModel


class RollingPCAEngine:
    """Walk-forward PCA factor engine.

    Re-estimates the PCA factor model on a trailing window every
    `rebalance_freq` trading days, then projects the OUT-OF-SAMPLE
    returns that follow (up to the next rebalance) onto the
    eigenvectors estimated as of the prior rebalance date.

    This is what makes the estimation genuinely rolling rather than
    static: on any given day, the factor space used to compute that
    day's factor return was fit using only data available strictly
    before that day. No future information leaks into the loadings
    used to score a given day.
    """

    def __init__(
        self,
        n_components: int = 5,
        window: int = 504,  # ~2 trading years
        rebalance_freq: int = 21,  # re-fit monthly by default
    ):
        if window <= 0:
            raise ValueError("window must be positive")
        if rebalance_freq <= 0:
            raise ValueError("rebalance_freq must be positive")

        self.n_components = n_components
        self.window = window
        self.rebalance_freq = rebalance_freq

        self.factor_returns_: pd.DataFrame | None = None
        self.loadings_history_: dict[pd.Timestamp, pd.DataFrame] = {}
        self.rebalance_dates_: list[pd.Timestamp] = []

    def fit_transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Walks forward through `returns`, re-fitting PCA every
        `rebalance_freq` days on the trailing `window` days of history.

        :param returns: Full-sample asset returns (T x N), time-indexed,
            sorted chronologically.
        :return: Out-of-sample factor return series covering every day
            for which a prior estimation window exists (i.e. the first
            `window` days have no factor return, since there is no
            history yet to estimate them from).
        """
        T = len(returns)
        if T <= self.window:
            raise ValueError(
                f"Need more than {self.window} observations to roll; got {T}."
            )

        self.loadings_history_.clear()
        self.rebalance_dates_.clear()
        chunks = []

        t = self.window
        while t < T:
            train = returns.iloc[t - self.window : t]
            oos_end = min(t + self.rebalance_freq, T)
            test = returns.iloc[t:oos_end]

            pca = PCAFactorModel(n_components=self.n_components).fit(train)

            rebalance_date = returns.index[t - 1]
            self.loadings_history_[rebalance_date] = pca.factor_loadings
            self.rebalance_dates_.append(rebalance_date)

            # Center the out-of-sample block with the TRAIN mean (not its
            # own mean) -- using the test block's own mean would leak
            # forward-looking information into today's factor return.
            X_test = test - train.mean()
            top_eigenvectors = pca.eigenvectors[:, : self.n_components]

            factor_chunk = pd.DataFrame(
                np.dot(X_test.to_numpy(), top_eigenvectors),
                index=test.index,
                columns=[f"PC_{i + 1}" for i in range(self.n_components)],
            )
            chunks.append(factor_chunk)

            t = oos_end

        self.factor_returns_ = pd.concat(chunks, axis=0)
        return self.factor_returns_

    def loadings_as_of(self, date) -> pd.DataFrame:
        """Returns the most recently estimated factor loadings as of `date`
        (the loadings that were actually live/tradeable on that date).
        """
        if not self.rebalance_dates_:
            raise ValueError("Engine has not been fit yet. Call fit_transform first.")

        eligible = [d for d in self.rebalance_dates_ if d <= date]
        if not eligible:
            raise ValueError(f"No estimation window available as of {date}.")

        return self.loadings_history_[max(eligible)]

    def latest_loadings(self) -> pd.DataFrame:
        """Returns the most recent factor loadings snapshot -- useful for
        feeding the current cluster/regime state to FactorClusterer without
        re-fitting the neural UMAP encoder on every single rebalance.
        """
        if not self.rebalance_dates_:
            raise ValueError("Engine has not been fit yet. Call fit_transform first.")
        return self.loadings_history_[self.rebalance_dates_[-1]]
