from sklearn.externals import joblib
import json

from time import time
from sklearn.manifold import TSNE

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from HelperFunctions.HelperFunction import HelperFunction
from HelperFunctions.Plotting import Plotting
from PrepareData.LoadData import LoadData
from Clustering.DBScanClustering import DBScanClustering
from Clustering.HDBScanClustering import HDBScanClustering


class Tsne:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil().get_config_instance()
        self.helper = HelperFunction()
        self.plot = Plotting()
        self.load_data = LoadData()
        self.dbscan = DBScanClustering()
        self.hdbscan = HDBScanClustering()

    def plot_matrix(self, Y, iteration, plot_path, init, perplexity):
        plt = self.plot.plot_it_2d(Y)
        plt.savefig(plot_path + "/" + "tsne_2d_" + str(iteration) + "_" + str(init) + "_" + str(perplexity) + ".png")
        plt.close()
        plt = self.plot.plot_it_3d(Y)
        plt.savefig(plot_path + "/" + "tsne_3d_" + str(iteration) + "_" + str(init) + "_" + str(perplexity) + ".png")
        plt.close()

    def perform_tsne(self, n_components, plot_path, model_path):
        start_time = time()
        model_list = list()
        reduced_matrix_list = list()
        perplexities = range(5, 55, 5)
        init_list = ["random", "pca"]
        
        for perplexity in perplexities:
            for init in init_list:
                model = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=1000, n_iter=5000, init=init)
                reduced_matrix = model.fit_transform(partial_matrix)
                model_list.append(model)
                reduced_matrix_list.append(reduced_matrix)
                self.plot_matrix(reduced_matrix, iteration, plot_path, init, perplexity)
                self.log.info("Saving 2d & 3d plots for init : {}, perplexity : {}".format(init, perplexity))

        return model_list, reduced_matrix_list

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
        start_time = time()
        plot_path = self.config['plots']['tsne']
        tsne_model_path = self.config['models']['tsne']
        tsne_results_path = self.config['results']['iterations']['tsne']
        pca_results_path = self.config["results"]["params"]["pca"]

        final_accuracies = dict()
        
        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        n_components = self.get_n_components(num_rows, pca_results_path)
        tsne_model_list, tsne_reduced_matrix_list = self.perform_tsne(n_components=n_components, plot_path=plot_path, model_path=tsne_model_path)
        self.log.info("Saving the TSNE model & Reduced Matrix at : {}".format(tsne_model_path))

        tsne_reduced_matrix_fname = tsne_model_path + "/" + "tsne_reduced_matrix_" + str(num_rows)
        np.savez_compressed(file=tsne_reduced_matrix_fname, arr=tsne_model_list)

        tsne_model_fname = tsne_model_path + "/" + "tsne_model_" + str(num_rows)
        np.savez_compressed(file=tsne_model_fname, arr=tsne_reduced_matrix_list)

        # TODO
        # Add clustering code.

        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    tsne = Tsne()
    tsne.main(num_rows=25000)