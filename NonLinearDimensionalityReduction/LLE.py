import json

from sklearn.externals import joblib
from sklearn.manifold import LocallyLinearEmbedding
from time import time

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from Clustering.DBScanClustering import DBScanClustering
from Clustering.HDBScanClustering import HDBScanClustering
from HelperFunctions.HelperFunction import HelperFunction
from HelperFunctions.Plotting import Plotting
from PrepareData.LoadData import LoadData


class LLE:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.config = ConfigUtil().get_config_instance()
        self.plot = Plotting()
        self.load_data = LoadData()
        self.dbscan = DBScanClustering()
        self.hdbscan = HDBScanClustering()

    def plot_matrix(self, reduced_matrix, iteration, plot_path, n_neighbors):
        plt = self.plot.plot_it_2d(reduced_matrix)
        plt.savefig(plot_path + "/" + "lle_2d_" + str(iteration) + "_" + str(n_neighbors) + ".png")
        plt.close()
        plt = self.plot.plot_it_3d(reduced_matrix)
        plt.savefig(plot_path + "/" + "lle_3d_" + str(iteration) + "_" + str(n_neighbors) + ".png")
        plt.close()

    def perform_lle(self, n_components, iteration, partial_matrix, partial_matrix_indices, plot_path, model_path):
        self.log.info("LLE on iteration #{}".format(iteration))
        n_neighbors_list = range(2, 22, 2)
        eigen_solver = "dense"
        eps_list = self.helper.frange(0.1, 1.1, 0.1)
        min_samples_list = range(2, 22, 2)
        min_cluster_size_list = range(2, 22, 2)

        dbscan_accuracies = dict()
        hdbscan_accuracies = dict()

        for n_neighbors in n_neighbors_list:
            model = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors,
                                           eigen_solver=eigen_solver, n_jobs=-1)
            self.log.info("Saving current model at : {}".format(model_path))
            reduced_matrix = model.fit_transform(partial_matrix.toarray())
            # Save the model in sklearn's joblib format.
            # joblib.dump(model, model_path + "/" + "lle_" + str(iteration) + "_" + str(n_neighbors) + ".model")
            self.plot_matrix(reduced_matrix, iteration, plot_path, n_neighbors)

            dbscan_accuracy_params = self.dbscan.dbscan_cluster(input_matrix=reduced_matrix,
                                                                input_matrix_indices=partial_matrix_indices,
                                                                eps_list=eps_list, min_samples_list=min_samples_list)

            hdbscan_accuracy_params = self.hdbscan.hdbscan_cluster(input_matrix=reduced_matrix,
                                                                   input_matrix_indices=partial_matrix_indices,
                                                                   min_cluster_size_list=min_cluster_size_list)

            key = "n_neighbors_" + str(n_neighbors)
            dbscan_accuracies[key] = dbscan_accuracy_params
            hdbscan_accuracies[key] = hdbscan_accuracy_params

        return dbscan_accuracies, hdbscan_accuracies

    def main(self):
        start_time = time()
        plot_path = self.config['plots']['lle']
        model_path = self.config['models']['lle']
        results_path = self.config['results']['lle']

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

            dbscan_accuracies, hdbscan_accuracies = self.perform_lle(n_components=3, iteration=index,
                                                                     partial_matrix=partial_matrix,
                                                                     partial_matrix_indices=partial_matrix_indices,
                                                                     plot_path=plot_path, model_path=model_path)

            self.log.info("DBScan Accuracy for Iteration #{} : {}".format(index, dbscan_accuracies))
            self.log.info("HDBScan Accuracy for Iteration #{} : {}".format(index, hdbscan_accuracies))
            final_accuracies[index] = [dbscan_accuracies, hdbscan_accuracies]
            index += 1

        json.dump(final_accuracies, open(results_path + "/" + "results_" + str(num_rows) + ".json", "w"))
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    lle = LLE()
    lle.main()
