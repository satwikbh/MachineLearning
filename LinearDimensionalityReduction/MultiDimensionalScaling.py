import json
from time import time

import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import pairwise_distances


from HelperFunctions.HelperFunction import HelperFunction
from LinearDimensionalityReduction.PrincipalComponentAnalysis import PrincipalComponentAnalysis
from PrepareData.LoadData import LoadData
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class MultiDimensionalScaling:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()
        self.helper = HelperFunction()
        self.pca = PrincipalComponentAnalysis()
        self.supported_dist_metrics = ['euclidean', 'cosine', 'jaccard', 'hamming']

    def perform_mds(self, n_components, input_matrix, seed):
        start_time = time()
        metrics = ['cosine', 'jaccard']
        self.log.info("The metrics enabled for MDS are : {}".format(metrics))
        mds = MDS(n_components=n_components, metric=True,
                  max_iter=900, eps=1e-12,
                  dissimilarity="precomputed",
                  random_state=seed, n_jobs=-1,
                  n_init=1)

        nmds = MDS(n_components=n_components, metric=False,
                   max_iter=900, eps=1e-12,
                   dissimilarity="precomputed",
                   random_state=seed, n_jobs=-1,
                   n_init=1)

        mds_reduced_matrix_list = list()
        nmds_reduced_matrix_list = list()

        for distance_metric in metrics:
            try:
                similarities = np.asarray([])

                if distance_metric in self.supported_dist_metrics:
                    similarities = pairwise_distances(input_matrix.toarray(), metric=distance_metric)
                else:
                    self.log.error("Distance metric : {} is not in the list of allowed values : {}".format(
                        distance_metric, 
                        self.supported_dist_metrics)
                    )

                similarities = self.pairwise_distance(distance_metric, input_matrix)

                self.log.info("Fitting MDS with {} distance metric".format(distance_metric))
                mds_reduced_matrix = mds.fit_transform(similarities)
                mds_reduced_matrix_list.append(mds_reduced_matrix)

                self.log.info("Fitting N-MDS with {} distance metric".format(distance_metric))
                nmds_reduced_matrix = nmds.fit_transform(similarities)
                nmds_reduced_matrix_list.append(nmds_reduced_matrix)
            except Exception as e:
                self.log.error("Error : {}".format(e))

        mds_reduced_matrix_list = np.asarray(mds_reduced_matrix_list)
        nmds_reduced_matrix_list = np.asarray(nmds_reduced_matrix_list)

        self.log.info("Time taken to perform_mds is {}".format(time() - start_time))
        return mds_reduced_matrix_list, nmds_reduced_matrix_list

    def get_n_components(self, num_rows, pca_results_path):
        fname = pca_results_path + "/" + "best_params_pca_" + str(num_rows) + ".json"
        if self.helper.is_file_present(fname):
            best_params = json.load(open(fname))
            best_param_value = min(best_params, key=best_params.get)
            n_components = best_param_value.split('n_components_')[1]
        else:
            n_components = self.pca.main(num_rows=num_rows, cluster_estimation=False)
        self.log.info("Number of components : {}".format(n_components))
        return int(n_components)

    def main(self, num_rows):
        """
        The main method.
        :param num_rows: Number of rows in the input matrix
        :return:
        """
        start_time = time()

        seed = np.random.RandomState(seed=3)

        mds_model_path = self.config["models"]["mds"]["model_path"]
        mds_results_path = self.config["results"]["iterations"]["mds"]
        pca_results_path = self.config["results"]["params"]["pca"]

        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        n_components = self.get_n_components(num_rows, pca_results_path)
        mds_reduced_matrix_list, nmds_reduced_matrix_list = self.perform_mds(n_components=n_components,
                                                                             input_matrix=input_matrix, seed=seed)
        self.log.info("Saving the MDS model at : {}".format(mds_model_path))
        mds_fname = mds_model_path + "/" + "mds_reduced_matrix_" + str(num_rows)
        nmds_fname = mds_model_path + "/" + "nmds_reduced_matrix_" + str(num_rows)
        np.savez_compressed(file=mds_fname, arr=mds_reduced_matrix_list)
        np.savez_compressed(file=nmds_fname, arr=nmds_reduced_matrix_list)

        # TODO
        # Add clustering code.

        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    mds = MultiDimensionalScaling()
    mds.main(num_rows=25000)
