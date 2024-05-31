from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
import umap
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Constants
DIM_REDUCTION_METHODS = {'umap': umap.UMAP, 'tsne': TSNE}
CLUSTER_METHODS = {'kmeans': KMeans, 'kmedoids': KMedoids}

def select_best_num_clusters(n_clusters: list, X: np.array, dim_reduction="umap", cluster_method="kmeans"):
    """
    Select the best number of clusters based on silhouette score.

    Parameters:
    - n_clusters: list of int, possible numbers of clusters to evaluate
    - X: np.array, the input data
    - dim_reduction: str, method for dimensionality reduction ('umap' or 'tsne')
    - cluster_method: str, clustering algorithm ('kmeans' or 'kmedoids')

    Returns:
    - results: dict, average silhouette scores for each number of clusters
    - best_n_clusters: int, the number of clusters with the highest average silhouette score
    """
    validate_inputs(dim_reduction, cluster_method)
    results = defaultdict()
    Clusterer = CLUSTER_METHODS[cluster_method]
    Reducer = DIM_REDUCTION_METHODS[dim_reduction]

    for cluster in n_clusters:
        clusterer = Clusterer(n_clusters=cluster, random_state=42) # init clustering algo
        labels = clusterer.fit_predict(X) # fit and predict input data
        silhouette_avg = round(silhouette_score(X=X, labels=labels), 5) # overall evaluation
        sample_silhouette_values = silhouette_samples(X, labels) # per-cluster evaluation
        print(f"Average silhouette score with {cluster} clusters: {silhouette_avg}")
        results[cluster] = silhouette_avg # save average silhouette score
        embeddings = Reducer().fit_transform(X) # dimensionality reduction
        plot_silhouette_coef(embeddings, cluster, silhouette_avg, sample_silhouette_values, labels, dim_reduction) # plotting results
        plt.suptitle(f"Silhouette analysis for {cluster_method.upper()} clustering k={cluster}", fontsize=14, fontweight="bold")
        plt.show()
    
    best_n_clusters = max(results, key=results.get) 
    return results, best_n_clusters # return overall results and optimal number of clusters

def validate_inputs(dim_reduction, cluster_method):
    if dim_reduction not in DIM_REDUCTION_METHODS:
        raise ValueError(f"dim_reduction should be one of {list(DIM_REDUCTION_METHODS.keys())}")
    if cluster_method not in CLUSTER_METHODS:
        raise ValueError(f"cluster_method should be one of {list(CLUSTER_METHODS.keys())}")

def plot_silhouette_coef(embeddings, cluster_number: int, silhouette_avg, sample_silhouette_values, labels, dim_reduction):
    """
    Plot the silhouette coefficients and the clusters.

    Parameters:
    - embeddings: np.array, the reduced dimensional embeddings of the data
    - cluster_number: int, number of clusters
    - silhouette_avg: float, average silhouette score
    - sample_silhouette_values: np.array, silhouette scores for each sample
    - labels: np.array, cluster labels for each sample
    """
    y_lower = 10
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 6)
    ax1.set_xlim([-0.5, 1]) # range of silhouette values
    
    for i in range(cluster_number):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / cluster_number)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title("The silhouette score overall and for each cluster")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.5, -0.25, -0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(labels.astype(float) / cluster_number)
    ax2.scatter(embeddings[:, 0], embeddings[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
    ax2.set_title(f"Visualization of clustered data: {dim_reduction.upper()}")