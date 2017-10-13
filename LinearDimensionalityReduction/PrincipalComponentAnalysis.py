import json
import pickle as pi
from time import time

import numpy as np
from sklearn.decomposition.pca import PCA

from Clustering.DBScanClustering import DBScanClustering
from Clustering.EstimateClusterParams import EstimateClusterParams
from Clustering.HDBScanClustering import HDBScanClustering
from HelperFunctions.HelperFunction import HelperFunction
from PrepareData.LoadData import LoadData
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class PrincipalComponentAnalysis:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()
        self.dbscan = DBScanClustering()
        self.hdbscan = HDBScanClustering()
        self.helper = HelperFunction()
        self.estimate_params = EstimateClusterParams()

    def perform_pca(self, input_matrix, num_rows, randomized=False):
        """
        Performs PCA. Here SVD will be solved using randomized algorithms.
        Function is not necessary explicitly as pca has svd_solver set to 'auto' by default which for higher dimension data performs randomized SVD.
        :param input_matrix: The input matrix ndarray form.
        :param num_rows: The number of rows.
        :param randomized: Perform randomized svd under the hood if true.
        Disabled by default.
        :return:
        """
        self.log.info("Entering the {} class".format(self.perform_pca.__name__))
        results_path = self.config['results']['params']['pca']
        reconstruction_error = self.config['models']['pca']['reconstruction_error']
        n_components_list = range(1000, 5100, 100)
        best_params = dict()
        for n_components in n_components_list:
            try:
                if randomized:
                    pca = PCA(n_components=n_components, svd_solver='randomized')
                else:
                    pca = PCA(n_components=n_components)
                reduced_matrix = pca.fit_transform(input_matrix)
                reconstructed_matrix = pca.inverse_transform(reduced_matrix)
                error_curr = self.helper.mean_square_error(reconstructed_matrix, input_matrix)
                self.log.info("Model for n_components : {}\tError : {}".format(n_components, error_curr))
                best_params['n_components_' + str(n_components)] = error_curr
                if error_curr * 100 < reconstruction_error:
                    json.dump(best_params, open(results_path + "/" + "best_params_pca_" + str(num_rows) + ".json", "w"))
                    break
            except Exception as e:
                self.log.error("Error : {}".format(e))
        self.log.info("Exiting the {} class".format(self.perform_pca.__name__))
        return reduced_matrix, n_components

    def main(self, num_rows, cluster_estimation=True):
        """
        The main method.
        :return:
        """
        start_time = time()
        pca_model_path = self.config["models"]["pca"]["model_path"]
        pca_results_path = self.config["results"]["iterations"]["pca"]
        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        reduced_matrix, n_components = self.perform_pca(input_matrix=input_matrix.toarray(), num_rows=num_rows, randomized=True)
        self.log.info("Saving the PCA model at : {}".format(pca_model_path))
        fname = pca_model_path + "/" + "pca_reduced_matrix_" + str(num_rows)
        np.save(file=fname, arr=reduced_matrix)

        if cluster_estimation is False:
            return n_components

        eps_list = self.helper.frange(0.1, 1.1, 0.1)
        min_samples_list = range(2, 22, 2)
        min_cluster_size_list = range(2, 22, 2)

        final_accuracies = dict()

        dbscan_accuracy_params = self.dbscan.dbscan_cluster(input_matrix=reduced_matrix,
                                                            input_matrix_indices=input_matrix_indices,
                                                            eps_list=eps_list,
                                                            min_samples_list=min_samples_list)

        hdbscan_accuracy_params = self.hdbscan.hdbscan_cluster(input_matrix=reduced_matrix,
                                                               input_matrix_indices=input_matrix_indices,
                                                               min_cluster_size_list=min_cluster_size_list)

        self.log.info("DBScan Accuracy : {}".format(dbscan_accuracy_params))
        self.log.info("HDBScan Accuracy : {}".format(hdbscan_accuracy_params))

        final_accuracies["dbscan"] = dbscan_accuracy_params
        final_accuracies["hdbscan"] = hdbscan_accuracy_params

        pi.dump(final_accuracies, open(pca_results_path + "/" + "pca_results_" + str(num_rows) + ".pickle", "w"))
        json.dump(final_accuracies, open(pca_results_path + "/" + "pca_results_" + str(num_rows) + ".json", "w"))

        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    pca = PrincipalComponentAnalysis()
    pca.main(num_rows=25000, cluster_estimation=True)
