import pickle as pi
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

    def perform_tsne(self, n_components, iteration, partial_matrix, partial_matrix_indices, plot_path, model_path):
        self.log.info("tsne on iteration #{}".format(iteration))
        perplexities = range(5, 55, 5)
        init = "random"
        eps_list = self.helper.frange(0.1, 1.0, 0.1)
        min_samples_list = range(2, 20, 2)
        min_cluster_size_list = range(2, 20, 2)

        dbscan_accuracies = dict()
        hdbscan_accuracies = dict()

        for perplexity in perplexities:
            model = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=1000, n_iter=5000, init=init)
            self.log.info("Saving current model at : {}".format(model_path))
            reduced_matrix = model.fit_transform(partial_matrix)
            pi.dump(model, open(model_path + "/" + "tsne_" + str(iteration) + "_" + str(perplexity) + ".model", "w"))
            self.plot_matrix(reduced_matrix, iteration, plot_path, init, perplexity)

            dbscan_accuracy_params = self.dbscan.dbscan_cluster(input_matrix=reduced_matrix,
                                                                input_matrix_indices=partial_matrix_indices,
                                                                eps_list=eps_list, min_samples_list=min_samples_list)

            hdbscan_accuracy_params = self.hdbscan.hdbscan_cluster(input_matrix=reduced_matrix,
                                                                   input_matrix_indices=partial_matrix_indices,
                                                                   min_cluster_size_list=min_cluster_size_list)

            key = "perplexity_" + str(perplexity)
            dbscan_accuracies[key] = dbscan_accuracy_params
            hdbscan_accuracies[key] = hdbscan_accuracy_params

        return dbscan_accuracies, hdbscan_accuracies

    def main(self):
        start_time = time()
        plot_path = self.config['plots']['tsne']
        model_path = self.config['models']['tsne']
        results_path = self.config['results']['tsne']

        num_rows = 25000
        batch_size = 1000
        counter = 0
        index = 0
        final_accuracies = dict()

        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        while counter < num_rows:
            partial_matrix = input_matrix[counter: counter + batch_size]
            partial_matrix_indices = input_matrix_indices[counter: counter + batch_size]
            counter += batch_size
            dbscan_accuracies, hdbscan_accuracies = self.perform_tsne(n_components=3, iteration=index,
                                                                      partial_matrix=partial_matrix,
                                                                      plot_path=plot_path,
                                                                      partial_matrix_indices=partial_matrix_indices,
                                                                      model_path=model_path)

            self.log.info("DBScan Accuracy for Iteration #{} : {}".format(index, dbscan_accuracies))
            self.log.info("HDBScan Accuracy for Iteration #{} : {}".format(index, hdbscan_accuracies))
            final_accuracies[index] = [dbscan_accuracies, hdbscan_accuracies]
            index += 1

        self.log.info("All the results are stored at : {}".format(results_path))
        pi.dump(final_accuracies, open(results_path + "/" + "results_" + str(num_rows) + ".pickle", "w"))
        json.dump(final_accuracies, open(results_path + "/" + "results_" + str(num_rows) + ".json", "w"))
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    tsne = Tsne()
    tsne.main()
