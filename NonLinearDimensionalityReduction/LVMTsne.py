from time import time

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from HelperFunctions.HelperFunction import HelperFunction
from HelperFunctions.Plotting import Plotting
from LinearDimensionalityReduction.PrincipalComponentAnalysis import PrincipalComponentAnalysis
from PrepareData.LoadData import LoadData
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class Tsne:
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
        learning_rate_list = range(100, 1100, 100)
        init_list = ["random", "pca"]

        for init in init_list:
            for perplexity in perplexity_list:
                for learning_rate in learning_rate_list:
                    model = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate,
                                 init=init)
                    model_list.append(model)
                    reduced_matrix = model.fit_transform(input_matrix.toarray())
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

    def get_best_params(self, tsne_model_list, tsne_reduced_matrix_list):
        self.log.info("Finding the best params for least error")
        main_list = list()
        for perplexity_params_ml in tsne_model_list:
            for learning_rate_params_ml in perplexity_params_ml:
                temp1 = learning_rate_params_ml['params']
                temp1['reduced_matrix'] = learning_rate_params_ml['reduced_matrix']
                main_list.append(temp1)
        df1 = pd.DataFrame(main_list)
        tsne_model = df1.loc[df1['kl_divergence'].idxmin()]['reduced_matrix']

        main_list = list()
        for perplexity_params_rml in tsne_reduced_matrix_list:
            for learning_rate_params_rml in perplexity_params_rml:
                temp2 = learning_rate_params_rml['params']
                temp2['reduced_matrix'] = learning_rate_params_rml['reduced_matrix']
                main_list.append(temp2)
        df2 = pd.DataFrame(main_list)
        tsne_reduced_matrix = df2.loc[df2['kl_divergence'].idxmin()]['reduced_matrix']

        return tsne_reduced_matrix, tsne_model

    def main(self, num_rows):
        start_time = time()
        plot_path = self.config['plots']['tsne']
        tsne_model_path = self.config['models']['tsne']
        tsne_results_path = self.config['results']['iterations']['tsne']
        n_components = 3

        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        tsne_model_list, tsne_reduced_matrix_list = self.perform_tsne(n_components=n_components,
                                                                      plot_path=plot_path,
                                                                      input_matrix=input_matrix)

        self.log.info("Saving the TSNE model & Reduced Matrix at : {}".format(tsne_model_path))

        all_params_dir = tsne_model_path + "/" + "all_params"
        self.helper.create_dir_if_absent(all_params_dir)

        tsne_all_reduced_matrix_fname = all_params_dir + "/" + "tsne_reduced_matrix_" + str(num_rows)
        np.savez_compressed(file=tsne_all_reduced_matrix_fname, arr=tsne_reduced_matrix_list)

        tsne_all_model_fname = all_params_dir + "/" + "tsne_model_" + str(num_rows)
        np.savez_compressed(file=tsne_all_model_fname, arr=tsne_model_list)

        best_tsne_reduced_matrix, best_tsne_model = self.get_best_params(tsne_model_list, tsne_reduced_matrix_list)

        tsne_reduced_matrix_fname = tsne_results_path + "/" + "tsne_reduced_matrix_" + str(num_rows)
        np.savez_compressed(file=tsne_reduced_matrix_fname, arr=best_tsne_reduced_matrix)

        tsne_model_fname = tsne_results_path + "/" + "tsne_model_" + str(num_rows)
        np.savez_compressed(file=tsne_model_fname, arr=best_tsne_model)

        # TODO
        # Add clustering code.

        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    tsne = Tsne()
    tsne.main(num_rows=25000)
