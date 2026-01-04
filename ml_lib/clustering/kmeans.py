import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-5):
        """
        n_clusters : number of clusters (K)
        max_iters  : maximum number of iterations
        tol        : tolerance for convergence
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    @staticmethod
    def euclidean_distance(point, centroids):
        """
        Compute distance between a point and all centroids
        """
        return np.sqrt(np.sum((centroids - point) ** 2, axis=1))

    def fit(self, X):
        """
        Train K-Means on data X
        X shape --> (n_samples, n_features)
        """
        n_samples, n_features = X.shape

        # 1️ Initialize centroids randomly within data range
        self.centroids = np.random.uniform(
            np.amin(X, axis=0),
            np.amax(X, axis=0),
            size=(self.n_clusters, n_features)
        )

        for _ in range(self.max_iters):
            # 2️ Assign clusters
            labels = []

            for point in X:
                distances = self.euclidean_distance(point, self.centroids)
                cluster_idx = np.argmin(distances)
                labels.append(cluster_idx)

            labels = np.array(labels)

            # 3️ Update centroids
            new_centroids = []

            for i in range(self.n_clusters):
                cluster_points = X[labels == i]

                if len(cluster_points) == 0:
                    # If cluster is empty, keep old centroid
                    new_centroids.append(self.centroids[i])
                else:
                    new_centroids.append(np.mean(cluster_points, axis=0))

            new_centroids = np.array(new_centroids)

            # 4️ Check convergence
            if np.max(np.abs(self.centroids - new_centroids)) < self.tol:
                break

            self.centroids = new_centroids

        return labels

    def predict(self, X):
        """
        Assign clusters to new data
        """
        labels = []

        for point in X:
            distances = self.euclidean_distance(point, self.centroids)
            cluster_idx = np.argmin(distances)
            labels.append(cluster_idx)

        return np.array(labels)

    







               