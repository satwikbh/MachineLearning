import time

import numpy as np
import scipy.cluster.hierarchy as hierarchy
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster

from Utils import LoggerUtil


class SingleLinkageClustering:
    def __init__(self):
        self.log = LoggerUtil.LoggerUtil(self.__class__.__name__).get()

    def single_linkage_clustering(self, inp, threshold, num_variants):
        self.log.info("************ SINGLE LINKAGE CLUSTERING *************")
        start_time = time.time()
        dataset = inp.astype(np.float32)

        # Centering the data
        dataset -= np.mean(dataset, axis=0)

        Z = hierarchy.linkage(dataset, method='ward', metric='euclidean')
        Z.dump(open("clustering_" + str(num_variants) + ".dump", "w"))
        self.log.info("Hierarchical Clustering : {}".format(Z))
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
            Z,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=12,  # show only the last p merged clusters
            show_leaf_counts=False,  # otherwise numbers in brackets are counts
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,  # to get a distribution impression in truncated branches
        )
        plt.show()

        last = Z[-10:, 2]
        last_rev = last[::-1]
        idxs = np.arange(1, len(last) + 1)
        plt.plot(idxs, last_rev)

        acceleration = np.diff(last, 2)  # 2nd derivative of the distances
        acceleration_rev = acceleration[::-1]
        plt.plot(idxs[:-2] + 1, acceleration_rev)
        plt.show()
        k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
        self.log.info("clusters: {}".format(k))

        clusters = fcluster(Z, threshold, criterion='maxclust')

        plt.figure(figsize=(10, 8))
        plt.scatter(dataset[:, 0], dataset[:, 1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
        plt.show()
        self.log.info("Time taken for SINGLE LINKAGE CLUSTERING : {}".format(time.time() - start_time))
