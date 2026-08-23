import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class FactorClusterer:

    def __init__(self, eps: float = 0.5, min_samples: int = 2):
        """Initializes DBSCAN clustering on PCA factor loadings.

        :param eps: The maximum distance between two samples for one to be
                    considered as in the neighborhood of the other.
        
        :param min_samples: The number of samples in a neighborhood for a point
                            to be considered as a core point.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = None
        self.clustered_assets_ = None

    def fit(self, factor_loadings: pd.DataFrame) -> "FactorClusterer":
        """Clusters assets dynamically based on their coordinates in PCA factor space.

        :param factor_loadings: DataFrame (N assets x K components) containing
        PCA loadings.
        """
        # Standardize factor loading coordinates
        scaler = StandardScaler()
        norm_loadings = scaler.fit_transform(factor_loadings.values)

        # Fit DBSCAN
        self.model.fit(norm_loadings)
        self.labels_ = self.model.labels_

        # Map labels back to asset tickers
        self.clustered_assets_ = pd.DataFrame(
            {"ticker": factor_loadings.index, "cluster": self.labels_}
        ).set_index("ticker")

        return self

    def get_clusters(self) -> dict[int, list[str]]:
        """Returns a dictionary mapping cluster IDs to lists of asset tickers.

        Note: Label -1 represents unclustered noise assets.
        """
        if self.clustered_assets_ is None:
            raise ValueError(
                "The model has not been fitted yet. Call fit() first."
            )

        clusters = {}
        for cluster_id in np.unique(self.labels_):
            tickers = self.clustered_assets_[
                self.clustered_assets_["cluster"] == cluster_id
            ].index.tolist()
            clusters[cluster_id] = tickers

        return clusters
