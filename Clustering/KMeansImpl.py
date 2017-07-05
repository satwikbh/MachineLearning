import json
import time

from collections import defaultdict
from sklearn.cluster import KMeans

from Utils.LoggerUtil import LoggerUtil


class KMeansImpl:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    def get_family_names(self, collection, clusters):
        entire_families = defaultdict(list)
        cluster = json.loads(clusters)
        for key, values in cluster.items():
            for each_malware in values:
                query = {"feature": "malheur", "key": each_malware}
                local_cursor = collection.find(query)
                for each_value in local_cursor:
                    key = each_value['key']
                    value = each_value['value'][key]["malheur"]["family"]
                    entire_families[value].append(key)
        for key, value in entire_families.items():
            self.log.info("Family : {} \t Variants : {}".format(key, len(value)))
        return json.dumps(entire_families)

    @staticmethod
    def core_model(input_matrix, k):
        model = KMeans(n_clusters=k)
        model.fit(input_matrix)
        return model.labels_

    def get_clusters_kmeans(self, input_matrix, names, k):
        self.log.info("************ KMeans Clustering Starts *************")
        start_time = time.time()
        cluster_label_list = self.core_model(input_matrix, k)
        d = defaultdict(list)
        for index, value in enumerate(cluster_label_list):
            d[str(value)].append(names[index])
        clusters = json.dumps(d)
        self.log.info("************ KMeans Clustering Ends *************")
        self.log.info("Total time taken : {}".format(time.time() - start_time))
        return clusters
