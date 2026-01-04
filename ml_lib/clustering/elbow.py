import numpy as np
import matplotlib.pyplot as plt
from .kmeans import KMeans

def elbow_method(X, max_k=10):
    wcss = []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit(X)
        centroids = kmeans.centroids

        # Compute WCSS manually
        total_wcss = 0
        for i in range(k):
            cluster_points = X[labels == i]
            total_wcss += np.sum((cluster_points - centroids[i]) ** 2)

        wcss.append(total_wcss)

    # Plot Elbow curve
    plt.figure(figsize=(8,5))
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.title("Elbow Method (Custom KMeans)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.grid(True)
    plt.show()

    return wcss