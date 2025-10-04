import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def load_dataset(path):
    if path and os.path.exists(path):
        print(f"Loading dataset from: {path}")
        df = pd.read_csv(path)
        print("Dataset shape:", df.shape)
        return df
    else:
        print("No valid dataset path provided — generating demo dataset.")
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
        df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
        return df


def pick_numeric_features(df, min_cols=2):
    ignore_cols = {'CustomerID', 'Customer Id', 'ID', 'Id', 'id', 'customer_id'}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ignore_cols]
    if len(num_cols) >= min_cols:
        selected = num_cols[:min_cols]
        print(f"Selected numeric columns: {selected}")
        return df[selected].copy()
    raise ValueError("Need at least 2 numeric columns in dataset.")


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def elbow_method(X, k_range=range(1, 11), outpath='outputs/elbow.png'):
    inertias = []
    ks = list(k_range)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, '-o')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    ensure_dir('outputs')
    plt.savefig(outpath)
    print(f"Saved elbow plot to {outpath}")
    plt.close()
    return ks, inertias


def silhouette_analysis(X, k_range=range(2, 11), outpath='outputs/silhouette.png'):
    scores = []
    ks = list(k_range)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
        print(f"Silhouette score for k={k}: {score:.4f}")
    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, '-o')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    ensure_dir('outputs')
    plt.savefig(outpath)
    print(f"Saved silhouette plot to {outpath}")
    plt.close()
    return ks, scores


def final_clustering(X_scaled, X_pca, k, df, outpath='outputs/clusters.png'):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    centroids_pca = PCA(n_components=2).fit(X_scaled).transform(km.cluster_centers_)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=30)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='black', marker='X', s=200, label='centroids')
    plt.legend()
    plt.title(f'KMeans Clusters (k={k})')
    ensure_dir('outputs')
    plt.savefig(outpath)
    print(f"Saved cluster plot to {outpath}")
    plt.close()

    df_out = df.copy().reset_index(drop=True)
    df_out['cluster'] = labels
    out_csv = 'outputs/clustered_data.csv'
    df_out.to_csv(out_csv, index=False)
    print(f"Saved clustered data to {out_csv}")

    sil = silhouette_score(X_scaled, labels)
    print(f"Final silhouette score for k={k}: {sil:.4f}")


def main(args):
    df = load_dataset(args.data)
    X_df = pick_numeric_features(df)
    X = X_df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    elbow_method(X_scaled)
    ks, scores = silhouette_analysis(X_scaled)

    if args.k and args.k > 1:
        best_k = args.k
    else:
        best_k = ks[int(np.argmax(scores))]
        print(f"Best k chosen by silhouette: {best_k}")

    final_clustering(X_scaled, X_pca, best_k, df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", help="Path to CSV dataset", default=None)
    parser.add_argument("--k", type=int, help="Number of clusters (optional)", default=None)
    args = parser.parse_args()

    main(args)

