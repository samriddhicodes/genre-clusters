# -*- coding: utf-8 -*-
"""
Created on Sun May 11 03:15:04 2025

@author: samri
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment  # Modern version

# --- K-means Functions ---
def dist(x, mu):
    return np.sum((x - mu) ** 2)

def assign_to_centroids(X, c, m, mu, centroids):
    K = centroids.shape[0]
    for i in range(m):
        distances = [dist(X[i], centroids[k]) for k in range(K)]
        min_index = np.argmin(distances)
        c[i] = min_index
        mu[i] = centroids[min_index]
    return mu, c

def J(X, centroids, c, m):
    total_distance = 0
    for j in range(m):
        total_distance += dist(X[j], centroids[int(c[j])])
    return total_distance / m

def find_new_centroids(X, m, n, K, centroids, c):
    new_centroids = np.zeros((K, n))
    for k in range(K):
        members = X[c == k]
        if len(members) > 0:
            new_centroids[k] = np.mean(members, axis=0)
        else:
            new_centroids[k] = np.random.uniform(low=X.min(axis=0), high=X.max(axis=0))
    return new_centroids

def kmeans(X, K, max_iters=100):
    m, n = X.shape
    centroids = np.random.uniform(low=X.min(axis=0), high=X.max(axis=0), size=(K, n))
    c = np.zeros(m, dtype=int)
    mu = np.zeros((m, n))

    for i in range(max_iters):
        mu, c = assign_to_centroids(X, c, m, mu, centroids)
        centroids = find_new_centroids(X, m, n, K, centroids, c)
        cost = J(X, centroids, c, m)
    return centroids, c, cost

# --- Load and preprocess the data ---
df = pd.read_csv("FinalProjectData/classical-pop-country-tracks.csv")
X = df.select_dtypes(include=[np.number]).iloc[:, :14].astype(float).to_numpy()
actual_labels = df["track_genre"].to_numpy()

m, n = X.shape
K = 3
centroids, c, cost = kmeans(X, K)

print("Updated Centroids:")
print(centroids)
print("Cost Function J =", cost)

# --- Optimal cluster-to-genre mapping using Hungarian Algorithm ---
genre_names = np.unique(actual_labels)
confusion = np.zeros((K, len(genre_names)), dtype=int)

for i in range(len(c)):
    cluster_id = c[i]
    genre_id = np.where(genre_names == actual_labels[i])[0][0]
    confusion[cluster_id, genre_id] += 1

row_ind, col_ind = linear_sum_assignment(-confusion)  # maximize matches
label_map = {row: genre_names[col] for row, col in zip(row_ind, col_ind)}

predicted_labels = np.array([label_map[cluster] for cluster in c])

print("Optimal Cluster-to-Genre Mapping:")
print(label_map)

# --- Evaluation ---
print("\nConfusion Matrix:")
print(confusion_matrix(actual_labels, predicted_labels))

print("\nClassification Report:")
print(classification_report(actual_labels, predicted_labels, zero_division=0))

# --- PCA Visualization ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(6,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=c, cmap='tab10')
plt.title("PCA Projection of Clustered Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
