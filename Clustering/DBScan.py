import pickle as pi

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

reduced_matrix = pi.load(open("ReducedMatrix.dump"))
print(reduced_matrix.shape)

eps_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
min_samples_list = [10, 20, 30, 40, 50]

for min_samples in min_samples_list:
    for eps in eps_list:
        X = StandardScaler().fit_transform(reduced_matrix)
        db = DBSCAN(eps=eps).fit(reduced_matrix)
        labels = db.labels_
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        print("Number of clusters : {}".format(n_clusters_))

        '''
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = 'k'

            class_member_mask = labels == k

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.savefig("dbscan_eps_" + str(eps) + "_" + "min_sample" + "_" + str(min_samples) + ".png")
        '''