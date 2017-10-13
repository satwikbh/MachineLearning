import json
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

    def perform_tsne(self, n_components, plot_path, input_matrix, pca_reduced_matrix):
        model_list = list()
        reduced_matrix_list = list()
        perplexity_list = range(5, 55, 5)
        init_list = ["random", "pca"]

        for init in init_list:
            if init == "pca":
                input_matrix = pca_reduced_matrix
            for perplexity in perplexity_list:
                model = TSNE(n_components=n_components, perplexity=perplexity,
                             init=init, n_jobs=-1, n_iter=1000)
                reduced_matrix = model.fit_transform(input_matrix.toarray())
                model_list.append(model)
                reduced_matrix_list.append(reduced_matrix)
                self.plot_matrix(reduced_matrix, plot_path, init, perplexity)
                self.log.info("Saving 2d plots")
                self.log.info("Model Params : \n"
                              "init : {}\t"
                              "perplexity : {}\t".format(init,
                                                         perplexity))

        return model_list, reduced_matrix_list

    def get_pca_reduced_matrix(self, num_rows, pca_results_path, pca_model_path):
        fname = pca_results_path + "/" + "best_params_pca_" + str(num_rows) + ".json"
        if self.helper.is_file_present(fname):
            best_params = json.load(open(fname))
            best_param_value = min(best_params, key=best_params.get)
            n_components = best_param_value.split('n_components_')[1]
        else:
            n_components = self.pca.main(num_rows=num_rows, cluster_estimation=False)
        self.log.info("Number of components : {}".format(n_components))
        fname = pca_model_path + "/" + "pca_reduced_matrix_" + str(num_rows)
        pca_reduced_matrix = np.load(file=fname)
        return pca_reduced_matrix

    def main(self, num_rows):
        start_time = time()
        plot_path = self.config['plots']['p_tsne']
        tsne_model_path = self.config['models']['p_tsne']
        tsne_results_path = self.config['results']['iterations']['p_tsne']

        pca_results_path = self.config["results"]["params"]["pca"]
        pca_model_path = self.config["models"]["pca"]["model_path"]
        n_components = 2

        final_accuracies = dict()

        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        pca_reduced_matrix = self.get_pca_reduced_matrix(num_rows, pca_results_path, pca_model_path)

        tsne_model_list, tsne_reduced_matrix_list = self.perform_tsne(n_components=n_components,
                                                                      plot_path=plot_path,
                                                                      input_matrix=input_matrix,
                                                                      pca_reduced_matrix=pca_reduced_matrix)

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
