import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from umap.parametric_umap import ParametricUMAP


class FactorClusterer:

    def __init__(
        self,
        n_components: int = 2,
        eps: float = 0.4,
        min_samples: int = 2,
        n_neighbors: int = 5,
        min_dist: float = 0.05,
        metric: str = "cosine",
        n_epochs: int = 200,
        random_state: int = 42,
    ):
        
        """Initializes Parametric UMAP non-linear manifold reduction coupled with DBSCAN.

        :param n_components: Target latent space dimensions (2 or 3 for
        optimal DBSCAN density).
        :param eps: Maximum radius for DBSCAN neighborhood evaluation in UMAP
        space.
        :param min_samples: Minimum asset count required to establish a core
        cluster.
        :param n_neighbors: Local metric neighborhood size for UMAP manifold
        construction.
        :param min_dist: Effective minimum distance between embedded points in
        latent space.
        :param metric: Distance metric for high-dimensional loadings (cosine
        aligns directional betas).
        :param n_epochs: Number of training epochs for the Parametric UMAP
        neural network.
        """
        
        self.n_components = n_components
        self.eps = eps
        self.min_samples = min_samples

        self.scaler = StandardScaler()

        # Neural Network parameterized UMAP reducer
        self.umap_reducer = ParametricUMAP(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_epochs=n_epochs,
            random_state=random_state,
            verbose=False,
        )

        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = None
        self.clustered_assets_ = None
        self.umap_embeddings_ = None

    def fit(self, factor_loadings: pd.DataFrame) -> "FactorClusterer":
        
        """Fits Parametric UMAP on PCA factor loadings and clusters assets via DBSCAN."""
        
        # 1. Standardize factor loadings
        norm_loadings = self.scaler.fit_transform(factor_loadings.values)

        # 2. Train neural network encoder and project to latent topological space
        self.umap_embeddings_ = self.umap_reducer.fit_transform(norm_loadings)

        # 3. Fit DBSCAN on low-dimensional, high-density UMAP coordinates
        self.dbscan.fit(self.umap_embeddings_)
        self.labels_ = self.dbscan.labels_

        # 4. Map cluster labels to tickers
        self.clustered_assets_ = pd.DataFrame(
            {
                "ticker": factor_loadings.index,
                "cluster": self.labels_,
                "umap_dim1": self.umap_embeddings_[:, 0],
                "umap_dim2": self.umap_embeddings_[:, 1],
            }
        ).set_index("ticker")

        return self

    def transform(self, new_factor_loadings: pd.DataFrame) -> np.ndarray:
        
        """Projects out-of-sample or rolling-window factor loadings into the learned

        latent space using the trained neural encoder.
        """
        
        norm_new = self.scaler.transform(new_factor_loadings.values)
        return self.umap_reducer.transform(norm_new)

    def get_clusters(self) -> dict[int, list[str]]:
        
        """Returns dictionary of cluster assignments, where -1 denotes unclustered noise."""
        
        if self.clustered_assets_ is None:
            raise ValueError(
                "Model has not been fitted. Call fit(factor_loadings) first."
            )

        clusters = {}
        for cluster_id in np.unique(self.labels_):
            tickers = self.clustered_assets_[
                self.clustered_assets_["cluster"] == cluster_id
            ].index.tolist()
            clusters[cluster_id] = tickers
        
        return clusters
