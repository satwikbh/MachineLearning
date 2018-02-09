import json
from time import time

import numpy as np
from sklearn.decomposition.kernel_pca import KernelPCA

from Clustering.DBScanClustering import DBScanClustering
from Clustering.HDBScanClustering import HDBScanClustering
from HelperFunctions.HelperFunction import HelperFunction
from LinearDimensionalityReduction.PrincipalComponentAnalysis import PrincipalComponentAnalysis
from PrepareData.LoadData import LoadData
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class KPCA:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()
        self.dbscan = DBScanClustering()
        self.hdbscan = HDBScanClustering()
        self.helper = HelperFunction()
        self.pca = PrincipalComponentAnalysis()

    def poly_kernel(self, **kwargs):
        degree_list = kwargs['degree_list']
        kernel = kwargs['kernel']
        n_components = kwargs['n_components']
        input_matrix = kwargs['input_matrix']
        best_params = kwargs['best_params']
        error_prev = kwargs['error_prev']
        reconstruction_error = kwargs['reconstruction_error']

        for degree in degree_list:
            self.log.info("Params are \t Kernel : {} \t Degree : {}".format(kernel, degree))
            kpca_model = KernelPCA(degree=degree, kernel=kernel, fit_inverse_transform=True,
                                   n_components=n_components, n_jobs=30)
            reduced_matrix = kpca_model.fit_transform(input_matrix)
            projected_matrix = kpca_model.inverse_transform(reduced_matrix)
            error_curr = self.helper.mean_square_error(input_matrix, projected_matrix)
            best_params[kernel]['degree_' + str(degree)] = error_curr
            if (error_curr - error_prev) * 100 < reconstruction_error:
                break
            error_prev = error_curr

    def kernel_pca(self, input_matrix, n_components, reconstruction_error, kpca_params_path):
        """
        Performs the Kernel Principal Component Reduction.
        :param input_matrix:
        :param n_components:
        :param reconstruction_error:
        :param kpca_params_path:
        :return:
        """
        self.log.info("Inside the {} class".format(self.kernel_pca.__name__))

        kernels = ['poly', 'rbf', 'sigmoid']
        gamma_list = self.helper.frange(0.0001, 0.0051, 0.0001)
        degree_list = self.helper.frange(1, 51, 1)

        best_params = dict()
        error_prev = 0.0
        for kernel in kernels:
            try:
                best_params[kernel] = {}
                if kernel is 'poly':
                    self.poly_kernel(degree_list=degree_list, kernel=kernel, n_components=n_components,
                                     input_matrix=input_matrix, best_params=best_params,
                                     reconstruction_error=reconstruction_error, error_prev=error_prev)
                if kernel in ['rbf', 'sigmoid']:
                    for gamma in gamma_list:
                        self.log.info("Params are \t Kernel : {} \t Gamma : {}".format(kernel, gamma))
                        kpca_model = KernelPCA(gamma=gamma, kernel=kernel, fit_inverse_transform=True,
                                               n_components=n_components, n_jobs=30)
                        reduced_matrix = kpca_model.fit_transform(input_matrix)
                        projected_matrix = kpca_model.inverse_transform(reduced_matrix)
                        error_curr = self.helper.mean_square_error(input_matrix, projected_matrix)
                        best_params[kernel]['gamma_' + str(gamma)] = error_curr
                        if (error_curr - error_prev) * 100 < reconstruction_error:
                            break
                        error_prev = error_curr
            except Exception as e:
                self.log.error("Error : {}".format(e))
        json.dump(best_params, open(kpca_params_path + "/" + "best_params_kpca.json", "w"))
        self.log.info("Exiting the {} class".format(self.kernel_pca.__name__))
        return reduced_matrix

    def get_n_components(self, num_rows, pca_results_path):
        fname = pca_results_path + "/" + "best_params_pca_" + str(num_rows) + ".json"
        if self.helper.is_file_present(fname):
            best_params = json.load(open(fname))
            best_param_value = min(best_params, key=best_params.get)
            n_components = best_param_value.split('n_components_')[1]
        else:
            n_components = self.pca.main(num_rows=num_rows)
        self.log.info("Number of components : {}".format(n_components))
        return n_components

    def main(self, num_rows):
        """
        The main method.
        :return:
        """
        start_time = time()
        kpca_model_path = self.config["models"]["kpca"]["model_path"]
        kpca_params_path = self.config['results']['params']['kpca']
        pca_results_path = self.config["results"]["params"]["pca"]

        reconstruction_error = self.config['models']['kpca']['reconstruction_error']

        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        n_components = self.get_n_components(num_rows, pca_results_path)
        reduced_matrix = self.kernel_pca(input_matrix=input_matrix.toarray(), n_components=n_components,
                                         reconstruction_error=reconstruction_error, kpca_params_path=kpca_params_path)
        self.log.info("Saving the K-PCA model at : {}".format(kpca_model_path))
        fname = kpca_model_path + "/" + "kpca_reduced_matrix_" + str(num_rows)
        np.save(file=fname, arr=reduced_matrix)

        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    kpca = KPCA()
    kpca.main(num_rows=2000)
