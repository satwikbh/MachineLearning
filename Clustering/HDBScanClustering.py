from time import time
from sklearn import metrics
from collections import defaultdict
from hdbscan import HDBSCAN

from Utils.LoggerUtil import LoggerUtil
from Clustering.AvclassValidation import AvclassValidation


class HDBScanClustering:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.labels = defaultdict(list)
        self.validation = AvclassValidation()

    def hdbscan_cluster(self, input_matrix, input_matrix_indices, min_cluster_size_list):
        """
        HDBScan algorithm implementation.
        It takes as input a list of min_cluster_size and the performs hdbscan and returns an list of accuracies for each.
        :param input_matrix: A input matrix in ndarray format.
        :param input_matrix_indices: These indices will be useful to compute the cluster accuracy.
        Since the dataset is distributed, giving the indices will help to locate the correct chunk.
        :param min_cluster_size_list: list of the minimum cluster size to consider when performing hdbscan
        :return:
        """
        self.log.info("************ HDDBSCAN Clustering started *************")
        start_time = time()
        accuracy_params = dict()
        for min_cluster_size in min_cluster_size_list:
            try:
                labels = HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=-1).fit_predict(input_matrix)
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters_ == 0:
                    self.log.info("min_cluster_size : {}\tNo of clusters inferred is : {}".format(
                        min_cluster_size,
                        len(labels)))
                    continue
                else:
                    cluster_accuracy = self.validation.main(labels=labels, input_matrix_indices=input_matrix_indices)
                    key = "min_cluster_size_" + str(min_cluster_size)
                    accuracy_params[key] = cluster_accuracy
                    silhouette_coefficient = metrics.silhouette_score(input_matrix, labels=labels)
                    self.log.info("min_cluster_size : {}\tNo of clusters : {}\t"
                                  "Silhouette Coeff : {}\tcluster accuracy : {}".format(min_cluster_size,
                                                                                        len(labels),
                                                                                        silhouette_coefficient,
                                                                                        cluster_accuracy))
            except Exception as e:
                self.log.error("Error : {}".format(e))
        self.log.info("************ HDDBSCAN Clustering started *************")
        self.log.info("Total time taken : {}".format(time() - start_time))
        return accuracy_params
