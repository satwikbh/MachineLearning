import json
import pickle as pi
import numpy as np
from time import time

from sklearn.decomposition.kernel_pca import KernelPCA

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from HelperFunctions.HelperFunction import HelperFunction
from PrepareData.LoadData import LoadData
from Clustering.DBScanClustering import DBScanClustering
from Clustering.HDBScanClustering import HDBScanClustering
from Clustering.EstimateClusterParams import EstimateClusterParams
from LinearDimensionalityReduction.PrincipalComponentAnalysis import PrincipalComponentAnalysis


class KPCA:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()
        self.dbscan = DBScanClustering()
        self.hdbscan = HDBScanClustering()
        self.helper = HelperFunction()
        self.estimate_params = EstimateClusterParams()
        self.pca = PrincipalComponentAnalysis()

    def kernel_pca(self, input_matrix, n_components):
        """
        Performs the Kernel Principal Component Reduction.
        :param input_matrix:
        :param n_components:
        :return:
        """
        global reduced_matrix
        self.log.info("Inside the {} class".format(self.kernel_pca.__name__))
        reconstruction_error = self.config['models']['kpca']['reconstruction_error']
        kernels = ['poly', 'rbf', 'sigmoid']
        gamma_list = self.helper.frange(0.0001, 0.0051, 0.0001)
        degree_list = self.helper.frange(1, 51, 1)
        results_path = self.config['results']['params']['kpca']
        best_params = dict()
        error_prev = 0.0
        for kernel in kernels:
            try:
                best_params[kernel] = {}
                if kernel is 'poly':
                    for degree in degree_list:
                        self.log.info("Params are \nKernel : {} \t Degree : {}".format(kernel, degree))
                        kpca = KernelPCA(degree=degree, kernel=kernel, fit_inverse_transform=True,
                                         n_components=n_components, n_jobs=30)
                        reduced_matrix = kpca.fit_transform(input_matrix)
                        projected_matrix = kpca.inverse_transform(reduced_matrix)
                        error_curr = self.helper.mean_square_error(input_matrix, projected_matrix)
                        best_params[kernel]['degree_' + str(degree)] = error_curr
                        if (error_curr - error_prev) * 100 < reconstruction_error:
                            break
                        error_prev = error_curr
                if kernel in ['rbf', 'sigmoid']:
                    for gamma in gamma_list:
                        self.log.info("Params are \nKernel : {} \t Gamma : {}".format(kernel, gamma))
                        kpca = KernelPCA(gamma=gamma, kernel=kernel, fit_inverse_transform=True,
                                         n_components=n_components,
                                         n_jobs=30)
                        reduced_matrix = kpca.fit_transform(input_matrix)
                        projected_matrix = kpca.inverse_transform(reduced_matrix)
                        error_curr = self.helper.mean_square_error(input_matrix, projected_matrix)
                        best_params[kernel]['gamma_' + str(gamma)] = error_curr
                        if (error_curr - error_prev) * 100 < reconstruction_error:
                            break
                        error_prev = error_curr
            except Exception as e:
                self.log.error("Error : {}".format(e))
        json.dump(best_params, open(results_path + "/" + "best_params_kpca.json", "w"))
        self.log.info("Exiting the {} class".format(self.kernel_pca.__name__))
        return reduced_matrix

    def get_n_components(self, num_rows, pca_results_path):
        fname = pca_results_path + "/" + "best_params_pca_" + str(num_rows) + ".json"
        if self.helper.is_file_present(fname):
            best_params = json.load(open(fname))
            best_param_value = min(best_params, key=best_params.get)
            n_components = best_param_value.split('n_components_')[1]
        else:
            n_components = self.pca.main(num_rows=num_rows, cluster_estimation=False)
        self.log.info("Number of components : {}".format(n_components))
        return n_components

    def main(self, num_rows):
        """
        The main method.
        :return:
        """
        start_time = time()
        kpca_model_path = self.config["models"]["kpca"]["model_path"]
        kpca_results_path = self.config["results"]["iterations"]["kpca"]
        pca_results_path = self.config["results"]["params"]["pca"]

        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        n_components = self.get_n_components(num_rows, pca_results_path)
        reduced_matrix = self.kernel_pca(input_matrix=input_matrix.toarray(), n_components=n_components)
        self.log.info("Saving the K-PCA model at : {}".format(kpca_model_path))
        fname = kpca_model_path + "/" + "kpca_reduced_matrix_" + str(num_rows)
        np.save(file=fname, arr=reduced_matrix)

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

        pi.dump(final_accuracies, open(kpca_results_path + "/" + "kpca_results_" + str(num_rows) + ".pickle", "w"))
        json.dump(final_accuracies, open(kpca_results_path + "/" + "kpca_results_" + str(num_rows) + ".json", "w"))

        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    kpca = KPCA()
    kpca.main(num_rows=25000)
