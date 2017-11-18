from time import time

import numpy as np
from sklearn.manifold import TSNE

from HelperFunctions.HelperFunction import HelperFunction
from HelperFunctions.Plotting import Plotting
from LinearDimensionalityReduction.PrincipalComponentAnalysis import PrincipalComponentAnalysis
from PrepareData.LoadData import LoadData
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class Tsne:
    """
    Grid Search is done in parallel code.
    The least error params are taken and this class generates the Reduced representation.
    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil().get_config_instance()
        self.helper = HelperFunction()
        self.plot = Plotting()
        self.load_data = LoadData()
        self.pca = PrincipalComponentAnalysis()

    def plot_matrix(self, reduced_matrix, plot_path, init, perplexity):
        plt = self.plot.plot_it_2d(reduced_matrix)
        plt.savefig(plot_path + "/" + "tsne_2d_" + str(init) + "_" + str(perplexity) + ".png")
        plt.close()
        plt = self.plot.plot_it_3d(reduced_matrix)
        plt.savefig(plot_path + "/" + "tsne_3d_" + str(init) + "_" + str(perplexity) + ".png")
        plt.close()

    def perform_tsne(self, n_components, plot_path, input_matrix, params_list, tsne_results_path, num_rows):
        for params in params_list:
            init = params["init"]
            perplexity = params["perplexity"]
            learning_rate = params["learning_rate"]
            model = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, init=init)
            reduced_matrix = model.fit_transform(input_matrix.toarray())
            self.plot_matrix(reduced_matrix, plot_path, init, perplexity)
            self.log.info("Saving 2d & 3d plots")
            self.log.info("Model Params : \n"
                          "init : {}\t"
                          "perplexity : {}\t"
                          "learning_rate : {}\t"
                          "kl_divergence : {}".format(init,
                                                      perplexity,
                                                      learning_rate,
                                                      model.kl_divergence_))
            self.log.info("Saving the TSNE Reduced Matrix at : {}".format(tsne_results_path))
            tsne_reduced_matrix_fname = tsne_results_path + "/" + "tsne_reduced_matrix_init_" + str(init) + \
                                        "_perplexity_" + str(perplexity) + \
                                        "_learning_rate_" + str(learning_rate) + str(num_rows)
            np.savez_compressed(file=tsne_reduced_matrix_fname, arr=reduced_matrix)

    @staticmethod
    def get_params():
        """
        Return the best params which were found in the ParallelTsne approach.
        :return:
        """
        params = list()
        inner_param = dict()
        inner_param["perplexity"] = 5
        inner_param["init"] = "random"
        inner_param["learning_rate"] = 100
        params.append(inner_param)

        inner_param["perplexity"] = 5
        inner_param["init"] = "pca"
        inner_param["learning_rate"] = 100
        params.append(inner_param)
        return params

    def main(self, num_rows):
        start_time = time()
        plot_path = self.config['plots']['tsne']
        tsne_results_path = self.config['results']['iterations']['tsne']
        n_components = 3

        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        params = self.get_params()
        self.perform_tsne(n_components=n_components,
                          plot_path=plot_path,
                          input_matrix=input_matrix,
                          params_list=params,
                          tsne_results_path=tsne_results_path,
                          num_rows=num_rows)

        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    tsne = Tsne()
    tsne.main(num_rows=50000)
