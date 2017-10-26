import json
import pickle as pi
import urllib
from collections import defaultdict
from time import time

import numpy as np
from sklearn.cluster import KMeans

from AvclassValidation import AvclassValidation
from ClusterMetrics import ClusterMetrics
from PrepareData.LoadData import LoadData
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from Utils.LoggerUtil import LoggerUtil


class KMeansImpl:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()
        self.metric = ClusterMetrics()
        self.validation = AvclassValidation()
        self.db_utils = DBUtils()

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

    def perform_kmeans(self, input_matrix, list_of_keys, avclass_collection):
        results_list = list()
        for k_value in range(2, 400):
            cluster_labels = self.core_model(input_matrix, k_value)
            cluster_accuracy, input_labels = self.validation.main(labels_pred=cluster_labels,
                                                                  list_of_keys=list_of_keys,
                                                                  avclass_collection=avclass_collection)
            s_score = self.metric.silhouette_score(input_matrix, cluster_labels)
            ch_score = self.metric.calinski_harabaz_score(input_matrix, cluster_labels)
            cluster_accuracy['s_score'] = s_score
            cluster_accuracy['ch_score'] = ch_score
            results_list.append(cluster_accuracy)
            self.log.info(cluster_accuracy)
        results_array = np.asarray(results_list)
        return results_array

    def prepare_kmeans(self, dr_matrices, list_of_keys, avclass_collection):
        dr_results_array = dict()
        for dr_name, dr_matrix in dr_matrices.items():
            if dr_name is "pca":
                self.log.info("Performing K-Means on PCA")
            elif dr_name is "mds":
                self.log.info("Performing K-Means on Metric MDS")
            elif dr_name is "nmds":
                self.log.info("Performing K-Means on Non-Metric MDS")
            elif dr_name is "tsne":
                self.log.info("Performing K-Means on TSNE")
            else:
                self.log.error("Dimensionality Reduction technique employed is not supported!!!")
            dr_results_array[dr_name] = self.perform_kmeans(dr_matrix, list_of_keys, avclass_collection)
        return dr_results_array

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
        tsne_reduced_matrix = np.load(tsne_file_name)
        dr_matrices["tsne"] = tsne_reduced_matrix

        return dr_matrices

    def save_results(self, num_rows, dr_results_array, pca_results_path, mds_results_path, tsne_results_path):
        for dr_name, dr_results in dr_results_array.items():
            if dr_name is "pca":
                pca_fname = pca_results_path + "kmeans_pca_" + str(num_rows)
                np.savez_compressed(pca_fname, dr_results)
            elif dr_name is "mds":
                mds_fname = mds_results_path + "kmeans_mds_" + str(num_rows)
                np.savez_compressed(mds_fname, dr_results)
            elif dr_name is "nmds":
                nmds_fname = mds_results_path + "kmeans_nmds_" + str(num_rows)
                np.savez_compressed(nmds_fname, dr_results)
            elif dr_name is "tsne":
                tsne_fname = tsne_results_path + "kmeans_tsne_" + str(num_rows)
                np.savez_compressed(tsne_fname, dr_results)
            else:
                self.log.error("Dimensionality Reduction technique employed is not supported!!!")

    def avclass_labeller(self, input_matrix_indices):
        client, avclass_collection = self.get_connection()
        names_path = self.config["data"]["list_of_keys"]
        temp = pi.load(open(names_path + "/" + "names.dump"))
        list_of_keys = list()

        for index in input_matrix_indices:
            try:
                val = temp[index]
                if "VirusShare" in val:
                    val = val.split("_")[1]
                list_of_keys.append(val)
            except Exception as e:
                self.log.error("Error : {}".format(e))
        return list_of_keys, avclass_collection

    def main(self, num_rows):
        start_time = time()

        pca_model_path = self.config["models"]["pca"]["model_path"]
        mds_model_path = self.config["models"]["mds"]["model_path"]
        tsne_model_path = self.config["models"]["tsne"]

        pca_results_path = self.config["results"]["iterations"]["pca"]
        mds_results_path = self.config["results"]["iterations"]["mds"]
        tsne_results_path = self.config["results"]["iterations"]["tsne"]
        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)

        list_of_keys, avclass_collection = self.avclass_labeller(input_matrix_indices)

        dr_matrices = self.get_dr_matrices(pca_model_path, mds_model_path, tsne_model_path, num_rows)
        dr_results_array = self.prepare_kmeans(dr_matrices, list_of_keys, avclass_collection)

        self.save_results(num_rows, dr_results_array, pca_results_path, mds_results_path, tsne_results_path)

        self.log.info("Time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    kmeans = KMeansImpl()
    kmeans.main(num_rows=25000)
