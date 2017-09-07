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

    def labels2clusters(self, labels, list_of_keys, variant_labels):
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

    def compute_accuracy(self, input_labels):
        cluster_dist = dict()
        for cluster_label, family_names in input_labels.items():
            try:
                new_family_names = [x for x in family_names if "SINGLETON" not in x]
                unique = len(set(new_family_names))
                if len(new_family_names) > 0:
                    cluster_dist[cluster_label] = 1.0 - (unique * 1.0 / len(new_family_names))
                else:
                    cluster_dist[cluster_label] = 1.0 - (unique * 1.0 / 10 ** 9)
            except Exception as e:
                self.log.error("Error : {}".format(e))
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
            try:
                val = temp[index]
                if "VirusShare" in val:
                    val = val.split("_")[1]
                list_of_keys.append(val)
            except Exception as e:
                self.log.error("Error : {}".format(e))

        variant_labels = self.prepare_labels(list_of_keys, avclass_collection)
        input_labels = self.labels2clusters(labels, list_of_keys, variant_labels)
        cluster_accuracy = self.compute_accuracy(input_labels)

        self.log.info("Total time taken : {}".format(time() - start_time))
        return (cluster_accuracy, input_labels)
