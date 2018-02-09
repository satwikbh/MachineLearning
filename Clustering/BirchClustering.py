import urllib
import numpy as np
import pickle as pi

from time import time
from collections import defaultdict
from sklearn.cluster import Birch
from scipy.spatial.distance import pdist, squareform

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from PrepareData.LoadData import LoadData
from HelperFunctions.HelperFunction import HelperFunction
from AvclassValidation import AvclassValidation


class BirchClustering:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()
        self.db_utils = DBUtils()
        self.helper = HelperFunction()
        self.validation = AvclassValidation()

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

    def prepare_labels(self, list_of_keys, collection):
        """
        Take the list of keys and find what the avclass label is inferred for it.
        :param list_of_keys:
        :param collection:
        :return variant_labels: a dict which contains md5 as key and the possible families as values.
        """
        variant_labels = defaultdict(list)
        cursor = collection.find({"md5": {"$in": list_of_keys}})

        for index, doc in enumerate(cursor):
            try:
                if index % 1000 == 0:
                    self.log.info("Iteration : #{}".format(index / 1000))
                key = doc["md5"]
                family = doc["avclass"]["result"]
                variant_labels[key].append(family)
            except Exception as e:
                self.log.error("Error : {}".format(e))
        return variant_labels

    @staticmethod
    def get_dr_matrices(pca_model_path, mds_model_path, tsne_model_path, sae_model_path, num_rows):
        """
        Takes the dimensionality reduction techniques model_path's, loads the matrices.
        Returns the matrices as a dict.
        :param pca_model_path:
        :param mds_model_path:
        :param tsne_model_path:
        :param sae_model_path:
        :param num_rows:
        :return:
        """
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

        tsne_random_file_name = tsne_model_path + "/" + "tsne_reduced_matrix_" + str(num_rows) + ".npz"
        tsne_random_reduced_matrix = np.load(tsne_random_file_name)['arr']
        dr_matrices["tsne_random"] = tsne_random_reduced_matrix

        tsne_pca_file_name = tsne_model_path + "/" + "tsne_reduced_matrix_" + str(num_rows) + ".npz"
        tsne_pca_reduced_matrix = np.load(tsne_pca_file_name)['arr']
        dr_matrices["tsne_pca"] = tsne_pca_reduced_matrix

        sae_file_name = sae_model_path + "/" + "sae_reduced_matrix_" + str(num_rows) + ".npz"
        sae_reduced_matrix = np.load(sae_file_name)['arr_0']
        dr_matrices['sae'] = sae_reduced_matrix

        return dr_matrices

    @staticmethod
    def core_model(input_matrix, branching_factor, threshold, n_clusters):
        birch_model = Birch(branching_factor=branching_factor, n_clusters=n_clusters, threshold=threshold,
                            compute_labels=True)
        cluster_labels = birch_model.fit_predict(input_matrix)
        return cluster_labels

    def perform_birch(self, input_matrix, list_of_keys, variant_labels):
        """
        Performs Birch Clustering and returns list of accuracies for each.
        :param variant_labels:
        :param list_of_keys:
        :param input_matrix: A input matrix in ndarray format.
        :return:
        """
        results_list = list()
        threshold_list = self.helper.frange(0.1, 1.1, 0.1)
        branching_factor_list = range(5, 105, 5)
        n_clusters_list = range(5, 105, 5)

        metric_list = ["euclidean", "jaccard", "cosine"]
        distance_matrices = dict()

        for metric in metric_list:
            self.log.info("Computing distance matrix for metric : {}".format(metric))
            distance_matrix = squareform(pdist(input_matrix, metric=metric))
            self.log.info("Distance matrix shape : {}".format(distance_matrix.shape))
            distance_matrices[metric] = distance_matrix

        for threshold in threshold_list:
            for branching_factor in branching_factor_list:
                for n_clusters in n_clusters_list:
                    try:
                        cluster_labels = self.core_model(input_matrix=input_matrix, branching_factor=branching_factor,
                                                         threshold=threshold, n_clusters=n_clusters)
                        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                        noise = len([x for x in cluster_labels if x == -1])
                        if n_clusters == 0:
                            self.log.info("threshold : {}\tbranching_factor : {}\tn_clusters : {}\n"
                                          "No of clusters inferred is : {}".format(threshold, branching_factor,
                                                                                   n_clusters,
                                                                                   len(cluster_labels)))
                            continue
                        else:
                            cluster_accuracy, input_labels = self.validation.main(labels_pred=cluster_labels,
                                                                                  list_of_keys=list_of_keys,
                                                                                  variant_labels=variant_labels,
                                                                                  input_matrix=input_matrix,
                                                                                  distance_matrices=distance_matrices)
                            cluster_accuracy['noise'] = noise
                            results_list.append(cluster_accuracy)
                            self.log.info(cluster_accuracy)
                    except Exception as e:
                        self.log.error("Error : {}".format(e))
        results_array = np.asarray(results_list)
        return results_array

    def prepare_birch(self, dr_matrices, list_of_keys, variant_labels):
        dr_results_array = dict()
        for dr_name, dr_matrix in dr_matrices.items():
            if dr_name is "pca":
                self.log.info("Performing Birch on PCA")
            elif dr_name is "mds":
                self.log.info("Performing Birch on Metric MDS")
            elif dr_name is "nmds":
                self.log.info("Performing Birch on Non-Metric MDS")
            elif dr_name is "tsne_random":
                self.log.info("Performing Birch on TSNE with random init")
            elif dr_name is "tsne_pca":
                self.log.info("Performing Birch on TSNE with pca init")
            elif dr_name is "sae":
                self.log.info("Performing Birch on SAE")
            else:
                self.log.error("Dimensionality Reduction technique employed is not supported!!!")
            dr_results_array[dr_name] = self.perform_birch(dr_matrix, list_of_keys, variant_labels)
        return dr_results_array

    def save_results(self, num_rows, dr_results_array, pca_results_path, mds_results_path, tsne_results_path,
                     sae_results_path):
        for dr_name, dr_results in dr_results_array.items():
            if dr_name is "pca":
                pca_fname = pca_results_path + "birch_pca_" + str(num_rows)
                np.savez_compressed(pca_fname, dr_results)
            elif dr_name is "mds":
                mds_fname = mds_results_path + "birch_mds_" + str(num_rows)
                np.savez_compressed(mds_fname, dr_results)
            elif dr_name is "nmds":
                nmds_fname = mds_results_path + "birch_nmds_" + str(num_rows)
                np.savez_compressed(nmds_fname, dr_results)
            elif dr_name is "tsne_random":
                tsne_random_fname = tsne_results_path + "birch_tsne_random" + str(num_rows)
                np.savez_compressed(tsne_random_fname, dr_results)
            elif dr_name is "tsne_pca":
                tsne_pca_fname = tsne_results_path + "birch_tsne_pca" + str(num_rows)
                np.savez_compressed(tsne_pca_fname, dr_results)
            elif dr_name is "sae":
                sae_fname = sae_results_path + "birch_sae_" + str(num_rows)
                np.savez_compressed(sae_fname, dr_results)
            else:
                self.log.error("Dimensionality Reduction technique employed is not supported!!!")

    def main(self, num_rows):
        start_time = time()

        pca_model_path = self.config["models"]["pca"]["model_path"]
        mds_model_path = self.config["models"]["mds"]["model_path"]
        tsne_model_path = self.config["models"]["tsne"]["model_path"]
        sae_model_path = self.config["models"]["sae"]["model_path"]

        pca_results_path = self.config["results"]["clustering_results"]["pca"]
        mds_results_path = self.config["results"]["clustering_results"]["mds"]
        tsne_results_path = self.config["results"]["clustering_results"]["tsne"]
        sae_results_path = self.config["results"]["clustering_results"]["sae"]

        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)

        list_of_keys, avclass_collection = self.avclass_labeller(input_matrix_indices)
        variant_labels = self.prepare_labels(list_of_keys, avclass_collection)

        dr_matrices = self.get_dr_matrices(pca_model_path=pca_model_path, mds_model_path=mds_model_path,
                                           tsne_model_path=tsne_model_path, sae_model_path=sae_model_path,
                                           num_rows=num_rows)
        dr_results_array = self.prepare_birch(dr_matrices, list_of_keys, variant_labels)

        self.save_results(num_rows=num_rows, dr_results_array=dr_results_array, pca_results_path=pca_results_path,
                          mds_results_path=mds_results_path, tsne_results_path=tsne_results_path,
                          sae_results_path=sae_results_path)

        self.log.info("Time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    birch = BirchClustering()
    birch.main(num_rows=25000)
