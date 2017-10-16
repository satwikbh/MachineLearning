from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics import silhouette_score

from Utils.LoggerUtil import LoggerUtil


class ClusterMetrics:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    @staticmethod
    def silhouette_score(input_matrix, labels):
        s_score = silhouette_score(input_matrix, labels)
        return s_score

    @staticmethod
    def nmi_score(labels_true, labels_pred):
        nmi_score = adjusted_mutual_info_score(labels_true, labels_pred)
        return nmi_score

    @staticmethod
    def ari_score(labels_true, labels_pred):
        ari_score = adjusted_rand_score(labels_true, labels_pred)
        return ari_score

    def compute_accuracy(self, input_labels):
        cluster_dist = dict()
        for cluster_label, family_names in input_labels.items():
            try:
                unique = len(set(family_names))
                if len(family_names) > 0:
                    cluster_dist[cluster_label] = 1.0 - (unique * 1.0 / len(family_names))
                else:
                    cluster_dist[cluster_label] = 1.0 - (unique * 1.0 / 10 ** 9)
            except Exception as e:
                self.log.error("Error : {}".format(e))
        return cluster_dist
