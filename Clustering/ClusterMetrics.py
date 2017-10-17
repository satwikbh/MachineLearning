from sklearn import metrics

from Utils.LoggerUtil import LoggerUtil


class ClusterMetrics:
    """
    Compute various cluster evaluation metrics.
    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    @staticmethod
    def ari_score(labels_true, labels_pred):
        ari_score = metrics.adjusted_rand_score(labels_true, labels_pred)
        return ari_score

    @staticmethod
    def nmi_score(labels_true, labels_pred):
        nmi_score = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
        return nmi_score

    @staticmethod
    def homogeneity_score(labels_true, labels_pred):
        homogeneity_score = metrics.homogeneity_completeness_v_measure(labels_pred=labels_pred, labels_true=labels_true)
        return homogeneity_score

    @staticmethod
    def fowlkes_mallow_score(labels_true, labels_pred):
        fm_score = metrics.fowlkes_mallows_score(labels_true=labels_true, labels_pred=labels_pred)
        return fm_score

    @staticmethod
    def silhouette_score(input_matrix, labels):
        s_score = metrics.silhouette_score(input_matrix, labels)
        return s_score

    @staticmethod
    def calinski_harabaz_score(input_matrix, labels):
        ch_score = metrics.calinski_harabaz_score(input_matrix, labels)
        return ch_score

    def cluster_purity(self, input_labels):
        cluster_dist = dict()
        for cluster_label, family_names in input_labels.items():
            try:
                if len(family_names) > 0:
                    unique = set(family_names)
                    purity = max([family_names.count(x) for x in unique]) * 1.0 / len(family_names)
                    cluster_dist[str(cluster_label)] = purity
                else:
                    cluster_dist[str(cluster_label)] = 0.0
            except Exception as e:
                self.log.error("Error : {}".format(e))
        return cluster_dist
