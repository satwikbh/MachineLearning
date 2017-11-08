import pickle as pi
import urllib
import numpy as np

from collections import defaultdict
from time import time
from sklearn import metrics
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

from AvclassValidation import AvclassValidation
from ClusterMetrics import ClusterMetrics
from HelperFunctions.HelperFunction import HelperFunction
from PrepareData.LoadData import LoadData
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from Utils.LoggerUtil import LoggerUtil


class DBScanClustering:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()
        self.labels = defaultdict(list)
        self.validation = AvclassValidation()
        self.metric = ClusterMetrics()
        self.db_utils = DBUtils()
        self.helper = HelperFunction()

    def get_connection(self):
        username = self.config['environment']['mongo']['username']
        pwd = self.config['environment']['mongo']['password']
        password = urllib.quote(pwd)
        address = self.config['environment']['mongo']['address']
        port = self.config['environment']['mongo']['port']
        auth_db = self.config['environment']['mongo']['auth_db']
        is_auth_enabled = self.config['environment']['mongo']['is_auth_enabled']

        client = self.db_utils.get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=is_auth_enabled,
                                          username=username, password=password)

        db_name = self.config['environment']['mongo']['db_name']
        avclass_collection_name = self.config['environment']['mongo']['avclass_collection_name']

        db = client[db_name]
        avclass_collection = db[avclass_collection_name]
        return client, avclass_collection

    @staticmethod
    def get_dr_matrices(pca_model_path, mds_model_path, tsne_model_path, num_rows):
        dr_matrices = dict()
        pca_file_name = pca_model_path + "/" + "pca_reduced_matrix_" + str(num_rows) + ".npy"
        pca_reduced_matrix = np.load(pca_file_name)
        dr_matrices["pca"] = pca_reduced_matrix

        mds_file_name = mds_model_path + "/" + "mds_reduced_matrix_" + str(num_rows) + ".npz"
        mds_reduced_matrix = np.load(mds_file_name)['arr'][0]
        dr_matrices["mds"] = mds_reduced_matrix

        nmds_file_name = mds_model_path + "/" + "nmds_reduced_matrix_" + str(num_rows) + ".npz"
        nmds_reduced_matrix = np.load(nmds_file_name)['arr'][0]
        dr_matrices["nmds"] = nmds_reduced_matrix

        tsne_file_name = tsne_model_path + "/" + "tsne_reduced_matrix_" + str(num_rows) + ".npz"
        tsne_reduced_matrix = np.load(tsne_file_name)['arr'][0]
        dr_matrices["tsne"] = tsne_reduced_matrix

        return dr_matrices

    @staticmethod
    def core_model(input_matrix, eps, min_samples):
        dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        cluster_labels = dbscan_model.fit_predict(input_matrix)
        return cluster_labels

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
        accuracy_params = dict()
        for eps in eps_list:
            for min_samples in min_samples_list:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(input_matrix)
                    labels = dbscan.labels_
                    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters_ == 0:
                        self.log.info(
                            "eps : {}\tmin_samples : {}\tNo of clusters inferred is : {}".format(eps, min_samples,
                                                                                                 n_clusters_))
                        continue
                    else:
                        cluster_accuracy = self.validation.main(labels_pred=labels,
                                                                input_matrix_indices=input_matrix_indices)
                        key = "eps_" + str(eps) + "_min_samples_" + str(min_samples)
                        accuracy_params[key] = cluster_accuracy
                        silhouette_coefficient = metrics.silhouette_score(input_matrix, labels)

                        self.log.info("eps : {}\t"
                                      "min_samples : {}\t"
                                      "No of clusters : {}\t"
                                      "Silhouette Coeff : {}".format(eps,
                                                                     min_samples,
                                                                     n_clusters_,
                                                                     silhouette_coefficient))
                except Exception as e:
                    self.log.error("Error : {}".format(e))

        self.log.info("************ DBSCAN Clustering completed *************")
        self.log.info("Time taken : {}".format(time() - start_time))
        return accuracy_params
