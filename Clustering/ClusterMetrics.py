import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y
from sklearn import metrics

from Utils.LoggerUtil import LoggerUtil


class ClusterMetrics:
    """
    Compute various cluster evaluation metrics.
    The dunn index and Davies Bouldin index are taken from
    https://github.com/jqmviegas/
    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.external_index = ExternalIndices()
        self.internal_index = InternalIndices()

    def ari_score(self, labels_true, labels_pred):
        ari_score = self.internal_index.ari_score(labels_pred=labels_pred, labels_true=labels_true)
        return ari_score

    def nmi_score(self, labels_true, labels_pred):
        nmi_score = self.internal_index.nmi_score(labels_true=labels_true, labels_pred=labels_pred)
        return nmi_score

    def homogeneity_score(self, labels_true, labels_pred):
        homogeneity_score = self.internal_index.homogeneity_score(labels_pred=labels_pred, labels_true=labels_true)
        return [x for x in homogeneity_score]

    def fowlkes_mallow_score(self, labels_true, labels_pred):
        fm_score = self.internal_index.fowlkes_mallow_score(labels_true=labels_true, labels_pred=labels_pred)
        return fm_score

    def calinski_harabaz_score(self, input_matrix, labels):
        ch_score = self.internal_index.calinski_harabaz_score(input_matrix=input_matrix, labels=labels)
        return ch_score

    def silhouette_score(self, input_matrix, labels):
        s_score = self.external_index.silhouette_score(input_matrix=input_matrix, labels=labels)
        return s_score

    def cluster_purity(self, input_labels):
        cluster_dist = self.external_index.cluster_purity(input_labels=input_labels, log=self.log)
        return cluster_dist

    def dunn_index(self, distances, labels):
        dunn_index_value = self.external_index.dunn_fast(distances=distances, labels=labels)
        return dunn_index_value

    def davis_bouldin_index(self, input_matrix, labels):
        davis_bouldin_index_value = self.external_index.davies_bouldin_index(input_matrix, labels)
        return davis_bouldin_index_value


class InternalIndices:
    def __init__(self):
        pass

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
        return [homogeneity_score]

    @staticmethod
    def fowlkes_mallow_score(labels_true, labels_pred):
        fm_score = metrics.fowlkes_mallows_score(labels_true=labels_true, labels_pred=labels_pred)
        return fm_score

    @staticmethod
    def calinski_harabaz_score(input_matrix, labels):
        ch_score = metrics.calinski_harabaz_score(input_matrix, labels)
        return ch_score


class ExternalIndices:
    def __init__(self):
        pass

    @staticmethod
    def delta(ck, cl):
        values = np.ones([len(ck), len(cl)]) * 10000

        for i in range(0, len(ck)):
            for j in range(0, len(cl)):
                values[i, j] = np.linalg.norm(ck[i] - cl[j])

        return np.min(values)

    @staticmethod
    def big_delta(ci):
        values = np.zeros([len(ci), len(ci)])

        for i in range(0, len(ci)):
            for j in range(0, len(ci)):
                values[i, j] = np.linalg.norm(ci[i] - ci[j])

        return np.max(values)

    def dunn(self, k_list):
        """
        Dunn index [CVI]
        :param k_list: list of np.arrays
            A list containing a numpy array for each cluster |c| = number of clusters
            c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
        :return:
        """
        deltas = np.ones([len(k_list), len(k_list)]) * 1000000
        big_deltas = np.zeros([len(k_list), 1])
        l_range = list(range(0, len(k_list)))

        for k in l_range:
            for l in (l_range[0:k] + l_range[k + 1:]):
                deltas[k, l] = self.delta(k_list[k], k_list[l])

            big_deltas[k] = self.big_delta(k_list[k])

        di = np.min(deltas) / np.max(big_deltas)
        return di

    @staticmethod
    def delta_fast(ck, cl, distances):
        values = distances[np.where(ck)][:, np.where(cl)]
        values = values[np.nonzero(values)]
        return np.min(values)

    @staticmethod
    def big_delta_fast(ci, distances):
        values = distances[np.where(ci)][:, np.where(ci)]
        return np.max(values)

    def dunn_fast(self, distances, labels):
        ks = np.sort(np.unique(labels))

        deltas = np.ones([len(ks), len(ks)]) * 1000000
        big_deltas = np.zeros([len(ks), 1])

        l_range = list(range(0, len(ks)))

        for k in l_range:
            for l in (l_range[0:k] + l_range[k + 1:]):
                deltas[k, l] = self.delta_fast((labels == ks[k]), (labels == ks[l]), distances)

            big_deltas[k] = self.big_delta_fast((labels == ks[k]), distances)

        di = np.min(deltas) / np.max(big_deltas)
        return di

    @staticmethod
    def big_s(x, center):
        len_x = len(x)
        total = 0

        for i in range(len_x):
            total += np.linalg.norm(x[i] - center)

        return total / len_x

    @staticmethod
    def check_number_of_labels(n_labels, n_samples):
        if not 1 < n_labels < n_samples:
            self.log.error("Number of labels is {}. " +
                "Valid values are 2 to n_samples - 1 (inclusive)".format(n_labels))

    def davies_bouldin_index(self, input_matrix, labels):
        """Compute the Davies Bouldin index.
        The index is defined as the ratio of within-cluster
        and between-cluster distances.
        Parameters
        ----------
        X : array-like, shape (``n_samples``, ``n_features``)
            List of ``n_features``-dimensional data points. Each row corresponds
            to a single data point.
        labels : array-like, shape (``n_samples``,)
            Predicted labels for each sample.
        Returns
        -------
        score : float
            The resulting Davies-Bouldin index.
        References
        ----------
        .. [1] `Davies, David L.; Bouldin, Donald W. (1979).
           "A Cluster Separation Measure". IEEE Transactions on
           Pattern Analysis and Machine Intelligence. PAMI-1 (2): 224-227`_
        """
        X = input_matrix
        X, labels = check_X_y(X, labels)
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        n_samples, _ = X.shape
        n_labels = len(le.classes_)

        self.check_number_of_labels(n_labels, n_samples)
        intra_dists = np.zeros(n_labels)
        centroids = np.zeros((n_labels, len(X[0])), np.float32)
        for k in range(n_labels):
            cluster_k = X[labels == k]
            mean_k = np.mean(cluster_k, axis=0)
            centroids[k] = mean_k
            intra_dists[k] = np.average(metrics.pairwise_distances(cluster_k, [mean_k]))

        centroid_distances = metrics.pairwise_distances(centroids)
        with np.errstate(divide='ignore', invalid='ignore'):
            if np.all((intra_dists[:, None] + intra_dists) == 0.0) or \
               np.all(centroid_distances == 0.0):
                return 0.0
            scores = (intra_dists[:, None] + intra_dists)/centroid_distances
            # remove inf values
            scores[scores == np.inf] = np.nan
        return np.mean(np.nanmax(scores, axis=1))

    @staticmethod
    def silhouette_score(input_matrix, labels):
        s_score = metrics.silhouette_score(input_matrix, labels)
        return s_score

    @staticmethod
    def cluster_purity(input_labels, log):
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
                log.error("Error : {}".format(e))
        return cluster_dist
