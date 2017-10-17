import json
from collections import defaultdict
from time import time

import numpy as np
from sklearn.cluster import KMeans

from AvclassValidation import AvclassValidation
from ClusterMetrics import ClusterMetrics
from PrepareData.LoadData import LoadData
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class KMeansImpl:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()
        self.metric = ClusterMetrics()
        self.validation = AvclassValidation()

    def get_family_names(self, collection, clusters):
        entire_families = defaultdict(list)
        cluster = json.loads(clusters)
        for index, values in cluster.items():
            for each_malware in values:
                query = {"feature": "malheur", "key": each_malware}
                local_cursor = collection.find(query)
                for each_value in local_cursor:
                    key = each_value['key']
                    value = each_value['value'][key]["malheur"]["family"]
                    entire_families[index].append(value)
        for key, value in entire_families.items():
            self.log.info("Family : {} \t Variants : {}".format(key, len(value)))
        return json.dumps(entire_families)

    @staticmethod
    def core_model(input_matrix, k):
        model = KMeans(n_clusters=k)
        labels = model.fit_predict(input_matrix)
        return labels

    def perform_kmeans(self, input_matrix, input_matrix_indices):
        results_list = list()
        for k_value in range(5, len(input_matrix_indices)):
            cluster_labels = self.core_model(input_matrix, k_value)
            cluster_accuracy, input_labels = self.validation.main(cluster_labels, input_matrix_indices)
            s_score = self.metric.silhouette_score(input_matrix, cluster_labels)
            cluster_accuracy['s_score'] = s_score
            results_list.append(cluster_accuracy)
            self.log.info(cluster_accuracy)
        results_array = np.asarray(results_list)
        return results_array

    @staticmethod
    def get_dr_matrices(pca_model_path, mds_model_path, tsne_model_path, num_rows):
        pca_file_name = pca_model_path + "/" + "pca_reduced_matrix_" + str(num_rows) + ".npy"
        pca_reduced_matrix = np.load(pca_file_name)

        mds_file_name = mds_model_path + "/" + "mds_reduced_matrix_" + str(num_rows) + ".npz"
        nmds_file_name = mds_model_path + "/" + "nmds_reduced_matrix_" + str(num_rows) + ".npz"

        mds_reduced_matrix = np.load(mds_file_name)
        nmds_reduced_matrix = np.load(nmds_file_name)

        tsne_file_name = tsne_model_path + "/" + "tsne_reduced_matrix_" + str(num_rows) + ".npz"
        tsne_reduced_matrix = np.load(tsne_file_name)

        return pca_reduced_matrix, mds_reduced_matrix, nmds_reduced_matrix, tsne_reduced_matrix

    @staticmethod
    def save_results(num_rows, pca_results_path, pca_results_array,
                     mds_results_path, mds_results_array, nmds_results_array,
                     tsne_results_path, tsne_results_array):
        pca_fname = pca_results_path + "kmeans_pca_" + str(num_rows)
        np.savez_compressed(pca_fname, pca_results_array)

        mds_fname = mds_results_path + "kmeans_mds_" + str(num_rows)
        np.savez_compressed(mds_fname, mds_results_array)

        nmds_fname = mds_results_path + "kmeans_nmds_" + str(num_rows)
        np.savez_compressed(nmds_fname, nmds_results_array)

        tsne_fname = tsne_results_path + "kmeans_tsne_" + str(num_rows)
        np.savez_compressed(tsne_fname, tsne_results_array)

    def main(self, num_rows):
        start_time = time()
        pca_model_path = self.config["models"]["pca"]["model_path"]
        mds_model_path = self.config["models"]["mds"]["model_path"]
        tsne_model_path = self.config["models"]["tsne"]

        pca_results_path = self.config["results"]["iterations"]["pca"]
        mds_results_path = self.config["results"]["iterations"]["mds"]
        tsne_results_path = self.config["results"]["iterations"]["tsne"]
        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)

        pca_reduced_matrix, mds_reduced_matrix, nmds_reduced_matrix, tsne_reduced_matrix = self.get_dr_matrices(
            pca_model_path, mds_model_path, tsne_model_path, num_rows)

        self.log.info("Performing K-Means on PCA")
        pca_results_array = self.perform_kmeans(pca_reduced_matrix, input_matrix_indices)

        self.log.info("Performing K-Means on Metric MDS")
        mds_results_array = self.perform_kmeans(mds_reduced_matrix, input_matrix_indices)

        self.log.info("Performing K-Means on Non-Metric MDS")
        nmds_results_array = self.perform_kmeans(nmds_reduced_matrix, input_matrix_indices)

        self.log.info("Performing K-Means on TSNE")
        tsne_results_array = self.perform_kmeans(tsne_reduced_matrix, input_matrix_indices)

        self.save_results(num_rows, pca_results_path, pca_results_array,
                          mds_results_path, mds_results_array, nmds_results_array,
                          tsne_results_path, tsne_results_array)

        self.log.info("Time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    kmeans = KMeansImpl()
    kmeans.main(num_rows=25000)
