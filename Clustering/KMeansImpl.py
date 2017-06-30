import json
import time
from sklearn.cluster import KMeans
from Utils.LoggerUtil import LoggerUtil


class KMeansImpl:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    @staticmethod
    def core_model(input_matrix, k):
        model = KMeans(n_clusters=k)
        model.fit(input_matrix)
        return model.labels_

    def get_clusters_kmeans(self, input_matrix, names, k):
        self.log.info("************ KMeans Clustering Starts *************")
        start_time = time.time()
        cluster_label_list = self.core_model(input_matrix, k)
        d = dict()
        for index, value in enumerate(cluster_label_list):
            if str(value) in d.keys():
                d[str(value)].append(names[index])
            else:
                d[str(value)] = []
        clusters = json.dumps(d)
        self.log.info("************ KMeans Clustering Ends *************")
        self.log.info("Total time taken : {}".format(time.time() - start_time))
        return clusters
