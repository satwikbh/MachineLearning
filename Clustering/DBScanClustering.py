from time import time
from sklearn import metrics
from collections import defaultdict
from sklearn.cluster import DBSCAN

from Utils.LoggerUtil import LoggerUtil
from Clustering.AvclassValidation import AvclassValidation


class DBScanClustering:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.labels = defaultdict(list)
        self.validation = AvclassValidation()

    def dbscan_cluster(self, input_matrix, input_matrix_indices, eps_list, min_samples_list):
        """
        The dbscan implementation. It takes as input a list of eps values and a list of min_sample values.
        Perform dbscan over all possible values and then return the highest cluster validity percentage.
        :param input_matrix: The input matrix in ndarray format.
        :param input_matrix_indices: These indices will be useful to compute the cluster accuracy.
        Since the dataset is distributed, giving the indices will help to locate the correct chunk.
        :param eps_list: the list of value which eps can take.
        :param min_samples_list:
        :return:
        """
        self.log.info("************ DBSCAN Clustering started *************")
        start_time = time()
        accuracy_params = list()
        for eps in eps_list:
            for min_samples in min_samples_list:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(input_matrix)
                n_clusters = dbscan.labels_

                cluster_accuracy = self.validation.main(labels=n_clusters, input_matrix_indices=input_matrix_indices)
                accuracy_params.append(cluster_accuracy)
                silhouette_coefficient = metrics.silhouette_score(input_matrix, n_clusters)

                self.log.info(
                    "eps : {}\tmin_samples : {}\tNo of clusters : {}\tSilhouette Coeff : {}\tcluster accuracy : {}".format(
                        eps,
                        min_samples,
                        len(n_clusters),
                        silhouette_coefficient,
                        cluster_accuracy))

        self.log.info("************ DBSCAN Clustering completed *************")
        self.log.info("Time taken : {}".format(time() - start_time))
        return accuracy_params