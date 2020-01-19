# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:33:41 2020

@author: jiraw
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Data Generation
def createDataPoints(centroid_loc, numSamples, clusterDeviation):
    X, y = make_blobs(n_samples=numSamples, centers=centroid_loc, cluster_std=clusterDeviation)
    X = StandardScaler().fit_transform(X)
    return X, y

centroid_loc = [[4, 3], [2, -1]]
X, y = createDataPoints(centroid_loc, 100, 0.5)

epsilon = 1
minimumSamples = 5
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels = db.labels_

#Distinguish outliers
# Firts, create an array of booleans using the labels from db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Number of clusters in labels, ignoring noise if present.
n_cluster_ = len(set(labels)) - (1 if -1 in labels else 0)

# Remove repetition in labels by turning it into a set.
unique_labels = set(labels)


colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    class_members_mask = (labels ==k)
    xy = X[class_members_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker=u'o', alpha=0.5)
    
    