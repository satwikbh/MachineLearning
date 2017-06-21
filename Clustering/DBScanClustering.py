from collections import defaultdict

from sklearn.cluster import DBSCAN

from Utils.LoggerUtil import LoggerUtil


class DBScan:
    def __init__(self):
        self.labels = defaultdict(list)
        self.log = LoggerUtil(self.__class__.__name__).get()

    def dbscan_cluster(self, input_matrix, threshold):
        self.log.info("************ DBSCAN Clustering started *************")
        eps = 1 - threshold
        dbscan = DBSCAN(eps=eps).fit(input_matrix)

        d = dbscan.labels_.tolist()

        for key, value in enumerate(d):
            self.labels[str(value)].append(key)

        self.log.info("************ DBSCAN Clustering started *************")
        return d
