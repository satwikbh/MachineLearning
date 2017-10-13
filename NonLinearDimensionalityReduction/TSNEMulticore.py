from time import time

import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE

from HelperFunctions.HelperFunction import HelperFunction
from HelperFunctions.Plotting import Plotting
from LinearDimensionalityReduction.PrincipalComponentAnalysis import PrincipalComponentAnalysis
from PrepareData.LoadData import LoadData
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class TSNEMulticore:
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

    def perform_tsne(self, n_components, plot_path, input_matrix):
        model_list = list()
        reduced_matrix_list = list()
        perplexity_list = range(5, 55, 5)
        learning_rate_list = range(10, 1100, 100)
        init_list = ["random", "pca"]

        for init in init_list:
            for perplexity in perplexity_list:
                for learning_rate in learning_rate_list:
                    model = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate,
                                 init=init, n_jobs=-1)
                    reduced_matrix = model.fit_transform(input_matrix.toarray())
                    model_list.append(model)
                    reduced_matrix_list.append(reduced_matrix)
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

        return model_list, reduced_matrix_list

    def main(self, num_rows):
        start_time = time()
        plot_path = self.config['plots']['p_tsne']
        tsne_model_path = self.config['models']['p_tsne']
        tsne_results_path = self.config['results']['iterations']['p_tsne']
        n_components = 2

        final_accuracies = dict()

        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        tsne_model_list, tsne_reduced_matrix_list = self.perform_tsne(n_components=n_components,
                                                                      plot_path=plot_path,
                                                                      input_matrix=input_matrix)

        self.log.info("Saving the TSNE model & Reduced Matrix at : {}".format(tsne_model_path))

        tsne_reduced_matrix_fname = tsne_model_path + "/" + "tsne_reduced_matrix_" + str(num_rows)
        np.savez_compressed(file=tsne_reduced_matrix_fname, arr=tsne_model_list)

        tsne_model_fname = tsne_model_path + "/" + "tsne_model_" + str(num_rows)
        np.savez_compressed(file=tsne_model_fname, arr=tsne_reduced_matrix_list)

        # TODO
        # Add clustering code.

        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    tsne = TSNEMulticore()
    tsne.main(num_rows=25000)
