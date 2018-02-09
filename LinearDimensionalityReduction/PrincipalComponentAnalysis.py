import json
from time import time

import numpy as np
from sklearn.decomposition.pca import PCA

from Clustering.DBScanClustering import DBScanClustering
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

    def perform_pca(self, input_matrix, num_rows, pca_params_path, reconstruction_error, randomized=False):
        """
        Performs PCA. SVD will be solved using randomized algorithms. Explicit specification of randomized is not
        necessary as pca has svd_solver set to 'auto' by default for high dimension data and performs randomized SVD.
        :param input_matrix: The input matrix ndarray form.
        :param num_rows: The number of rows.
        :param pca_params_path:
        :param reconstruction_error:
        :param randomized: Perform randomized svd under the hood if true.
        Disabled by default.
        :return:
        """
        self.log.info("Entering the {} class".format(self.perform_pca.__name__))
        n_components_list = range(1000, 5100, 100)
        best_params = dict()
        for n_components in n_components_list:
            try:
                if randomized:
                    pca_model = PCA(n_components=n_components, svd_solver='randomized')
                else:
                    pca_model = PCA(n_components=n_components)
                reduced_matrix = pca_model.fit_transform(input_matrix)
                reconstructed_matrix = pca_model.inverse_transform(reduced_matrix)
                error_curr = self.helper.mean_square_error(reconstructed_matrix, input_matrix)
                self.log.info("Model for n_components : {}\tReconstruction Error : {}".format(n_components, error_curr))
                best_params['n_components_' + str(n_components)] = str(error_curr)
                if error_curr * 100 < reconstruction_error:
                    json.dump(best_params, open(pca_params_path + "/" + "best_params_pca_" + str(num_rows) + ".json", "w"))
                    break
            except Exception as e:
                self.log.error("Error : {}".format(e))
        self.log.info("Exiting the {} class".format(self.perform_pca.__name__))
        return reduced_matrix, n_components

    def main(self, num_rows):
        """
        The main method.
        :return:
        """
        start_time = time()
        pca_model_path = self.config["models"]["pca"]["model_path"]
        pca_params_path = self.config["results"]["params"]["pca"]
        reconstruction_error = self.config['models']['pca']['reconstruction_error']

        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        reduced_matrix, n_components = self.perform_pca(input_matrix=input_matrix.toarray(), num_rows=num_rows,
                                                        randomized=True, pca_params_path=pca_params_path,
                                                        reconstruction_error=reconstruction_error)
        self.log.info("Saving the PCA model at : {}".format(pca_model_path))
        fname = pca_model_path + "/" + "pca_reduced_matrix_" + str(num_rows)
        np.save(file=fname, arr=reduced_matrix)
        self.log.info("Total time taken : {}".format(time() - start_time))
        return n_components


if __name__ == '__main__':
    pca = PrincipalComponentAnalysis()
    pca.main(num_rows=25000)
