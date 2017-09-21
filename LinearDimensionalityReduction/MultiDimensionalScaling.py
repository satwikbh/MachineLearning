import json

from time import time

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from PrepareData.LoadData import LoadData


class DiscriminantAnalysis:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()

    def main(self):
        start_time = time()
        plot_path = self.config['plots']['lle']
        model_path = self.config['models']['lle']
        results_path = self.config['results']['iterations']['lle']

        num_rows = 25000
        batch_size = 1000
        counter = 0
        index = 0
        final_accuracies = dict()

        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        while counter < num_rows:
            self.log.info("Discriminant Analysis on iteration #{}".format(index))
            partial_matrix = input_matrix[counter: counter + batch_size]
            partial_matrix_indices = input_matrix_indices[counter: counter + batch_size]
            counter += batch_size
            da_accuracies = dict()

            dbscan_accuracies, hdbscan_accuracies = self.lda.perform__lda(n_components=3, iteration=index,
                                                                          partial_matrix=partial_matrix,
                                                                          partial_matrix_indices=partial_matrix_indices,
                                                                          plot_path=plot_path, model_path=model_path)

            da_accuracies["lda"] = [dbscan_accuracies, hdbscan_accuracies]

            dbscan_accuracies, hdbscan_accuracies = self.qda.perform_qda(n_components=3, iteration=index,
                                                                         partial_matrix=partial_matrix,
                                                                         partial_matrix_indices=partial_matrix_indices,
                                                                         plot_path=plot_path, model_path=model_path)

            dbscan_accuracies['qda'] = [dbscan_accuracies, hdbscan_accuracies]

            self.log.info("DBScan Accuracy for Iteration #{} : {}".format(index, dbscan_accuracies))
            self.log.info("HDBScan Accuracy for Iteration #{} : {}".format(index, hdbscan_accuracies))
            final_accuracies[index] = dbscan_accuracies
            index += 1

        json.dump(final_accuracies, open(results_path + "/" + "results_" + str(num_rows) + ".json", "w"))
        self.log.info("Total time taken : {}".format(time() - start_time))
