import json
from time import time

import numpy as np
from sklearn.manifold import isomap

from HelperFunctions.HelperFunction import HelperFunction
from LinearDimensionalityReduction.PrincipalComponentAnalysis import PrincipalComponentAnalysis
from PrepareData.LoadData import LoadData
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class Isomap:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()
        self.pca = PrincipalComponentAnalysis()

    def perform_isomap(self, n_neighbors_list, n_components, input_matrix):
        """
        Performs ISOMAP on the input matrix and returns a dict of reconstruction error and reduced matrix.
        :param n_neighbors_list: possible number of neighbors.
        :param n_components: number of eigen vectors to keep.
        :param input_matrix: the input matrix of shape (n_samples, n_features).
        :return iterations: dict with key, value being reconstruction error, reduced matrix.
        """
        start_time = time()
        iterations = dict()
        for n_neighbors in n_neighbors_list:
            self.log.info("Isomap for n_neighbors : {}".format(n_neighbors))
            model = isomap.Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=-1)
            reduced_matrix = model.fit_transform(input_matrix)
            reconstruction_error = model.reconstruction_error()
            iterations[reconstruction_error] = reduced_matrix
        self.log.info("Time taken for perform_isomap : {}".format(time() - start_time))
        return iterations

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
        start_time = time()

        pca_results_path = self.config["results"]["params"]["pca"]
        isomap_model_path = self.config["models"]["isomap"]["model_path"]
        isomap_results_path = self.config["results"]["iterations"]["isomap"]

        n_neighbors_list = range(2, 21, 1)

        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        n_components = self.get_n_components(num_rows, pca_results_path)
        reduced_matrix = self.perform_isomap(input_matrix=input_matrix.toarray(), n_neighbors_list=n_neighbors_list,
                                             n_components=n_components)
        self.log.info("Saving the ISOMAP model at : {}".format(isomap_model_path))
        fname = isomap_model_path + "/" + "isomap_reduced_matrix_" + str(num_rows)
        np.save(file=fname, arr=reduced_matrix)

        # TODO
        # Add clustering code.

        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    isomap = Isomap()
    isomap.main(num_rows=25000)
