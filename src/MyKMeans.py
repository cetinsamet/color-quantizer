#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MyKMeans.py
#
# Created by Samet Cetin.
# Contact: cetin.samet@outlook.com
#

import numpy as np
from scipy.spatial.distance import euclidean, sqeuclidean
from collections import Counter

EPSILON = 0.0001


class MyKMeans:
    """K-Means clustering similar to sklearn 
    library but different.
    https://goo.gl/bnuM33

    But still same.

    Parameters
    ----------
    n_clusters : int, optional, default: 3
        The number of clusters to form as well as the number of
        centroids to generate.
    init_method : string, optional, default: 'random'
        Initialization method. Values can be 'random', 'kmeans++'
        or 'manual'. If 'manual' then cluster_centers need to be set.
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    cluster_centers : np.array, used only if init_method is 'manual'.
        If init_method is 'manual' without fitting, these values can be used
        for prediction.
    """

    def __init__(self, init_method="random", n_clusters=3, max_iter=300, random_state=None, cluster_centers=[]):
        self.init_method    = init_method
        self.n_clusters     = n_clusters
        self.max_iter       = max_iter

        if type(random_state) == int:                       # CHECK IF random_state IS AN integer
            self.random_state   = np.random.RandomState(random_state)
        elif type(random_state) == np.random.RandomState:   # CHECK IF random_state IS A np.random.RandomState
            self.random_state   = random_state
        else:                                               # CHECK IF random_state IS None
            self.random_state = np.random

        if init_method == "manual":
            self.cluster_centers = cluster_centers
        else:
            self.cluster_centers = []

    def initialize(self, X):
        """ Initialize centroids according to self.init_method
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.
        Returns
        ----------
        self.cluster_centers : array-like, shape=(n_clusters, n_features)
        """

        # RANDOM METHOD ---------------------------------------------------------------------------------------------- #
        if self.init_method == 'random':
            n_samples, n_features   = X.shape
            random_centroid_indices = self.random_state.permutation(n_samples)[:self.n_clusters]
            self.cluster_centers    = X[random_centroid_indices].copy().astype(np.float64)
            return self.cluster_centers
        # -----------------------------------------------------------------------------------------------------------  #

        # KMEANS++ METHOD -------------------------------------------------------------------------------------------- #
        if self.init_method == 'kmeans++':
            n_samples, n_features   = X.shape
            centers                 = np.empty((self.n_clusters, n_features))

            # SELECT FIRST CENTROID RANDOMLY ------------------------------------------------------------------------- #
            center_id               = self.random_state.randint(n_samples)
            centers[0]              = X[center_id].copy()
            X = np.delete(X, center_id, axis=0)         # REMOVE SELECTED DATA POINT
            # -------------------------------------------------------------------------------------------------------- #

            # SELECT OTHER CENTROIDS BY APPLYING KMEANS++ ------------------------------------------------------------ #
            for c in range(1, self.n_clusters):

                max_dist, center_id = -1, -1
                for x_id, x_ in enumerate(X):
                    dist = sum([euclidean(x_, centers[c_id]) for c_id in range(c)])

                    if dist > max_dist:
                        max_dist    = dist
                        center_id   = x_id

                centers[c] = X[center_id].copy()
                X = np.delete(X, center_id, axis=0)     # REMOVE SELECTED DATA POINT
            # -------------------------------------------------------------------------------------------------------- #

            self.cluster_centers = centers.copy().astype(np.float64)
            return self.cluster_centers
        # -----------------------------------------------------------------------------------------------------------  #

        # MANUAL METHOD ---------------------------------------------------------------------------------------------- #
        if self.init_method == 'manual':
            return np.vstack(self.cluster_centers).astype(np.float64)
        # ------------------------------------------------------------------------------------------------------------ #

    def fit(self, X):

        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.
        Returns
        ----------
        self : MyKMeans
        """

        n_samples, n_features   = X.shape
        current_centroids       = self.cluster_centers.copy()

        # PERFORM ITERATION (MAX_ITER) ------------------------------------------------------------------------------- #
        for epoch in range(self.max_iter):

            # ASSIGN LABELS TO DATA POINTS --------------------------------------------------------------------------- #
            data_labels = [np.argmin([euclidean(x_, centroid) for centroid in current_centroids]) for x_ in X]
            # -------------------------------------------------------------------------------------------------------- #

            # CALCULATE NEW CENTROIDS -------------------------------------------------------------------------------- #
            next_centroids  = np.zeros((self.n_clusters, n_features))
            for i, label in enumerate(data_labels):
                next_centroids[label] += X[i]

            for k, v in Counter(data_labels).items():
                next_centroids[k] /= np.float64(v)

            #  CHECK HALTING CONDITION ------------------------------------------------------------------------------- #
            DIFF_CENTROID       = sum([euclidean(a, b) for a, b in zip(current_centroids, next_centroids)])
            current_centroids   = next_centroids.copy()

            if DIFF_CENTROID <= EPSILON:
                break
            # -------------------------------------------------------------------------------------------------------- #
        # ------------------------------------------------------------------------------------------------------------ #

        self.labels             = np.array(data_labels)
        self.cluster_centers    = current_centroids
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return np.array([np.argmin([euclidean(x_, centroid) for centroid in self.cluster_centers]) for x_ in X])

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels


if __name__ == "__main__":

    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])

    kmeans = MyKMeans(n_clusters=2, random_state=0, init_method='kmeans++')
    print kmeans.initialize(X)
    # [[4. 4.]
    #  [1. 0.]]

    kmeans = MyKMeans(n_clusters=2, random_state=0, init_method='random')
    print kmeans.initialize(X)
    # [[4. 0.]
    #  [1. 0.]]

    kmeans.fit(X)
    print kmeans.labels
    # array([1, 1, 1, 0, 0, 0])

    print kmeans.predict([[0, 0], [4, 4]])
    # array([1, 0])

    print kmeans.cluster_centers
    # array([[4, 2],
    #       [1, 2]])