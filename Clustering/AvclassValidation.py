from collections import defaultdict
from time import time

from ClusterMetrics import ClusterMetrics
from HelperFunctions.HelperFunction import HelperFunction
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from Utils.LoggerUtil import LoggerUtil


class AvclassValidation:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.db_utils = DBUtils()
        self.helper = HelperFunction()
        self.metrics = ClusterMetrics()

    @staticmethod
    def cursor2list(cursor):
        list_of_keys = list()
        for each in cursor:
            list_of_keys.append(each["_id"])
        return list_of_keys

    def labels2clusters(self, labels, list_of_keys, variant_labels):
        """
        Takes the labels and prepares clusters.
        All the md5 for a label are taken and their family is inferred and a dict is created out of them
        :param labels:
        :param list_of_keys:
        :param variant_labels:
        :return:
        """
        clusters = defaultdict(list)
        for index, value in enumerate(labels):
            try:
                md5 = list_of_keys[index]
                cluster_label = variant_labels[md5]
                clusters[value] += cluster_label
            except Exception as e:
                self.log.error("Error : {}".format(e))
        return clusters

    def prepare_labels(self, list_of_keys, collection):
        """
        Take the list of keys and find what the avclass label is inferred for it.
        :param list_of_keys:
        :param collection:
        :return variant_labels: a dict which contains md5 as key and the possible families as values.
        """
        variant_labels = defaultdict(list)
        cursor = collection.find({"md5": {"$in": list_of_keys}})

        for index, doc in enumerate(cursor):
            try:
                if index % 1000 == 0:
                    self.log.info("Iteration : #{}".format(index / 1000))
                key = doc["md5"]
                family = doc["avclass"]["result"]
                variant_labels[key].append(family)
            except Exception as e:
                self.log.error("Error : {}".format(e))
        return variant_labels

    def get_true_labels(self, variant_labels):
        labels_true = []
        labels = self.helper.flatten_list(variant_labels.values())
        unique_labels = set(labels)

        for index, label in enumerate(unique_labels):
            labels_true += [index] * labels.count(label)

        return labels_true

    def main(self, labels_pred, list_of_keys, avclass_collection):
        """
        The labels are sent as input.
        The output is each cluster with its accuracy and labels.
        :param list_of_keys:
        :param avclass_collection:
        :param labels_pred: The labels format is (cluster_label, malware_source).
        malware_source is found in database and usually starts with VirusShare_.
        Since the dataset is distributed, giving the indices will help to locate the correct chunk.
        :return: A dict which contains the a cluster label and the accuracy it has over all the variants.
        """
        start_time = time()
        variant_labels = self.prepare_labels(list_of_keys, avclass_collection)
        input_labels = self.labels2clusters(labels_pred, list_of_keys, variant_labels)
        labels_true = self.get_true_labels(variant_labels)

        ari_score = self.metrics.ari_score(labels_true=labels_true, labels_pred=labels_pred)
        nmi_score = self.metrics.nmi_score(labels_true=labels_true, labels_pred=labels_pred)
        homogeneity_score = self.metrics.homogeneity_score(labels_true=labels_true, labels_pred=labels_pred)
        fw_score = self.metrics.fowlkes_mallow_score(labels_true=labels_true, labels_pred=labels_pred)
        acc_score = self.metrics.cluster_purity(input_labels)

        cluster_accuracy = dict()
        cluster_accuracy['ari'] = ari_score
        cluster_accuracy['nmi'] = nmi_score
        cluster_accuracy['homogeneity_score'] = homogeneity_score
        cluster_accuracy['fw_score'] = fw_score
        cluster_accuracy['purity'] = acc_score

        self.log.info("Total time taken : {}".format(time() - start_time))
        return cluster_accuracy, input_labels
