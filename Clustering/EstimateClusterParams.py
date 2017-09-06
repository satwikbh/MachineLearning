from collections import OrderedDict

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil


class EstimateClusterParams:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()

    def dbscan_stats(self, dbscan):
        """
        Takes the dbscan values and returns the parameters for which the noise is least.
        :param dbscan:
        :return:
        """
        best_accuracy = dict()
        for n_neighbors in dbscan.keys():
            try:
                eps_min_sample_dict = dbscan[n_neighbors]
                self.log.info("n_neighbors : {}".format(n_neighbors))
                for eps_min_sample in eps_min_sample_dict.keys():
                    try:
                        accuracy, labels = eps_min_sample_dict[eps_min_sample]
                        self.log.info("eps_min_sample : {}".format(eps_min_sample))
                        for each_cluster in labels:
                            try:
                                labels_list = labels[each_cluster]
                                labels_list = [x for x in labels_list if "SINGLETON" not in x]
                                d = {x: labels_list.count(x) for x in labels_list}
                                d = OrderedDict(sorted(d.items(), key=lambda x: x[1]))
                                accuracy[each_cluster] = (max(d.values()) * 1.0 / len(labels_list)) * 100
                            except Exception as e:
                                self.log.error("cluster : {} \t error : {} ".format(each_cluster, e))

                        if len(best_accuracy.keys()) == 0:
                            best_accuracy["eps_min_sample"] = eps_min_sample
                            best_accuracy["n_neighbors"] = n_neighbors
                            if "-1" in accuracy:
                                best_accuracy["accuracy"] = accuracy["-1"]
                            else:
                                best_accuracy["accuracy"] = 0.0

                        if "-1" in accuracy and accuracy["-1"] < best_accuracy["accuracy"]:
                            best_accuracy["accuracy"] = accuracy["-1"]
                            best_accuracy["eps_min_sample"] = eps_min_sample
                            best_accuracy["n_neighbors"] = n_neighbors

                        print("accuracy : {}".format(accuracy))
                    except Exception as e:
                        self.log.error("eps_min_sample : {} \t error : {} ".format(eps_min_sample, e))
            except Exception as e:
                self.log.error("n_neighbors : {} \t error : {} ".format(n_neighbors, e))
        return best_accuracy

    def hdbscan_stats(self, hdbscan):
        """
        Takes the hdbscan values and returns the parameters for which the noise accuracy is least.
        :param hdbscan:
        :return:
        """
        # Here the best accuracy implies least noise accuracy.
        best_accuracy = dict()
        for n_neighbors in hdbscan.keys():
            try:
                min_cluster_size_list = hdbscan[n_neighbors]
                self.log.info("n_neighbors : {}".format(n_neighbors))
                for min_cluster_size in min_cluster_size_list:
                    try:
                        accuracy, labels = min_cluster_size_list[min_cluster_size]
                        self.log.info("min_cluster_size : {}".format(min_cluster_size))
                        for each_cluster in labels:
                            try:
                                labels_list = labels[each_cluster]
                                labels_list = [x for x in labels_list if "SINGLETON" not in x]
                                d = {x: labels_list.count(x) for x in labels_list}
                                d = OrderedDict(sorted(d.items(), key=lambda x: x[1]))
                                accuracy[each_cluster] = (max(d.values()) * 1.0 / len(labels_list)) * 100
                            except Exception as e:
                                self.log.error("cluster : {} \t error : {}".format(each_cluster, e))

                        if len(best_accuracy.keys()) == 0:
                            best_accuracy["min_cluster_size"] = min_cluster_size
                            best_accuracy["n_neighbors"] = n_neighbors
                            if "-1" in accuracy:
                                best_accuracy["accuracy"] = accuracy["-1"]
                            else:
                                best_accuracy["accuracy"] = 0.0

                        if "-1" in accuracy and accuracy["-1"] < best_accuracy["accuracy"]:
                            best_accuracy["accuracy"] = accuracy["-1"]
                            best_accuracy["min_cluster_size"] = min_cluster_size
                            best_accuracy["n_neighbors"] = n_neighbors

                    except Exception as e:
                        self.log.error("min_cluster_size : {} \t error : {}".format(min_cluster_size, e))
            except Exception as e:
                self.log.error("n_neighbors : {} \t error : {}".format(n_neighbors, e))
        return best_accuracy

    def main(self, results):
        results = self.config['results']['lle']
        best_params = dict()
        for chunk in results.keys():
            dbscan, hdbscan = results[chunk]
            dbscan_best_params = self.dbscan_stats(dbscan)
            hdbscan_best_params = self.hdbscan_stats(hdbscan)
            best_params[chunk]["dbscan"] = dbscan_best_params
            best_params[chunk]["hdbscan"] = hdbscan_best_params
        return best_params


if __name__ == '__main__':
    estimate = EstimateClusterParams()
    estimate.main()
