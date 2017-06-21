import json

from sklearn.cluster import KMeans


class KMeansImpl:
    def __init__(self):
        pass

    @staticmethod
    def run_k_means(input_matrix, k):
        model = KMeans(n_clusters=k)
        model.fit(input_matrix)
        return model.labels_

    @staticmethod
    def get_clusters_kmeans(input_matrix, names, k):
        cluster_label_list = KMeansImpl.run_k_means(input_matrix, k)
        d = dict()
        for index, value in enumerate(cluster_label_list):
            if str(value) in d.keys():
                d[str(value)].append(names[index])
            else:
                d[str(value)] = []
        clusters = json.dumps(d)
        return clusters
