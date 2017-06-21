from sklearn.cluster import Birch

from Utils.LoggerUtil import LoggerUtil


class BirchClustering:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    def birch_clustering(self, input_matrix, threshold):
        self.log.info("************ Birch Clustering started *************")
        brc = Birch(branching_factor=100, threshold=0.7, compute_labels=True, n_clusters=None)

        brc.fit(input_matrix)
        clusters_birch = brc.predict(input_matrix).tolist()
        self.log.info("Number of clusters : {}".format(len(set(clusters_birch))))

        self.log.info("************ Birch Clustering Ended *************")
        return clusters_birch
