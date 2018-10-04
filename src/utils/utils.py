from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import numpy as np

import keras.backend as K

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

class OutlierDetector(object):
    def __init__(self, ancs, n_clust):
        self.ancs = ancs
        km = KMeans(n_clusters=n_clust)
        self.labels = km.fit_predict(ancs)
        self.centers = km.cluster_centers_
        self.mean_dist = self.mean_cluster_distances(
            ancs, self.labels, self.centers)

    def mean_cluster_distances(self, preds, labels, centers):
        mean_dist = {}
        for ix in range(len(centers)):
            pred_cut = preds[np.where(labels==ix)]
            dist = 0
            for item in pred_cut:
                dist += euclidean(item, centers[ix])
            mean_dist[ix] = dist / len(pred_cut)
        return mean_dist

    def determine_connection(self, preds):
        distances = self.mean_dist.values()
        rv = []
        for i ,item in enumerate(preds):
            for center in self.centers:
                eu_d = euclidean(center, item)
                for distance in distances:
                    marker = False
                    if eu_d <= distance:
                        marker = True
            rv.append(marker)
        return rv