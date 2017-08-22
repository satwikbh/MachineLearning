import urllib
import pickle as pi

from collections import defaultdict
from time import time
from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils


class AvclassValidation:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.db_utils = DBUtils()

    def get_connection(self):
        username = self.config['environment']['mongo']['username']
        pwd = self.config['environment']['mongo']['password']
        password = urllib.quote(pwd)
        address = self.config['environment']['mongo']['address']
        port = self.config['environment']['mongo']['port']
        auth_db = self.config['environment']['mongo']['auth_db']
        is_auth_enabled = self.config['environment']['mongo']['is_auth_enabled']

        client = self.db_utils.get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=is_auth_enabled,
                                          username=username, password=password)

        db_name = self.config['environment']['mongo']['db_name']
        avclass_collection_name = self.config['environment']['mongo']['avclass_collection_name']

        db = client[db_name]
        avclass_collection = db[avclass_collection_name]
        return client, avclass_collection

    @staticmethod
    def cursor2list(cursor):
        list_of_keys = list()
        for each in cursor:
            list_of_keys.append(each["_id"])
        return list_of_keys

    @staticmethod
    def labels2clusters(labels):
        clusters = defaultdict(list)
        for cluster_label, name in enumerate(labels):
            clusters[cluster_label].append(name)
        return clusters

    def prepare_labels(self, list_of_keys, collection):
        variant_labels = defaultdict(list)
        cursor = collection.find({"md5": {"$in": list_of_keys}})

        for index, doc in enumerate(cursor):
            if index % 1000 == 0:
                self.log.info("Iteration : #{}".format(index / 1000))
            key = doc["md5"]
            family = doc["avclass"]["result"]
            variant_labels[key].append(family)

        return variant_labels

    @staticmethod
    def compute_accuracy(labels, variant_labels):
        list_of_clusters = list()
        for names in labels:
            list_of_clusters.append(variant_labels[names])

        total_no_of_values = len(list_of_clusters)

        cluster_dist = dict()
        for each_label in set(list_of_clusters):
            cluster_dist[each_label] = list_of_clusters.count(each_label) * 1.0 / total_no_of_values

        return cluster_dist

    def main(self, labels, input_matrix_indices):
        """
        The labels are sent as input.
        The output is each cluster with its accuracy and labels.
        :param labels: The labels format is (cluster_label, malware_source).
        malware_source is found in database and usually starts with VirusShare_.
        :param input_matrix_indices: These indices will be useful to compute the cluster accuracy.
        Since the dataset is distributed, giving the indices will help to locate the correct chunk.
        :return: A dict which contains the a cluster label and the accuracy it has over all the variants.
        """
        start_time = time()
        client, avclass_collection = self.get_connection()
        names_path = self.config["data"]["list_of_keys"]
        temp = pi.load(open(names_path + "/" + "names.dump"))
        list_of_keys = list()

        for index in input_matrix_indices:
            val = temp[index].split("_")[1]
            list_of_keys.append(val)

        variant_labels = self.prepare_labels(list_of_keys, avclass_collection)
        input_labels = self.labels2clusters(labels)

        cluster_accuracy = defaultdict(list)

        for key, value in input_labels:
            accuracy = self.compute_accuracy(value, variant_labels)
            cluster_accuracy[key].append(accuracy)

        self.log.info("Total time taken : {}".format(time() - start_time))
        return cluster_accuracy
