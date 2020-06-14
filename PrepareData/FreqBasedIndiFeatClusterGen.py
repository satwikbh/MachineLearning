import json
from time import time
from urllib.parse import quote

import pickle as pi
import numpy as np

from HelperFunctions.HelperFunction import HelperFunction
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from Utils.LoggerUtil import LoggerUtil


class FreqBasedIndiFeatClusterGen:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.db_utils = DBUtils()
        self.helper = HelperFunction()
        self.files_pool = None
        self.reg_keys_pool = None
        self.mutex_pool = None
        self.exec_commands_pool = None
        self.network_pool = None
        self.static_feature_pool = None
        self.stat_sign_pool = None

    def get_collection(self):
        username = self.config['environment']['mongo']['username']
        pwd = self.config['environment']['mongo']['password']
        password = quote(pwd)
        address = self.config['environment']['mongo']['address']
        port = self.config['environment']['mongo']['port']
        auth_db = self.config['environment']['mongo']['auth_db']
        is_auth_enabled = self.config['environment']['mongo']['is_auth_enabled']

        client = self.db_utils.get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=is_auth_enabled,
                                          username=username, password=password)

        db_name = self.config['environment']['mongo']['db_name']
        db = client[db_name]

        c2db_collection_name = self.config['environment']['mongo']['c2db_collection_name']
        c2db_collection = db[c2db_collection_name]

        return c2db_collection

    def get_list_of_keys(self, c2db_collection):
        cursor = c2db_collection.aggregate([{"$group": {"_id": "$key"}}])
        list_of_keys = self.helper.cursor_to_list(cursor, identifier="_id")
        return list_of_keys

    @staticmethod
    def get_cluster_dict():
        cluster_dict = dict()
        return cluster_dict

    @staticmethod
    def add_to_dict(curr_doc, feature_pool):
        for _ in curr_doc:
            if _ in feature_pool:
                feature_pool[_] += 1
            else:
                feature_pool[_] = 1

    @staticmethod
    def get_query(list_of_keys, feature):
        """
        This method ensures the order of keys for the in query.
        :param list_of_keys:
        :param feature:
        :return:
        """
        query = [
            {"$match": {"key": {"$in": list_of_keys}, "feature": feature}},
            {"$addFields": {"__order": {"$indexOfArray": [list_of_keys, "$key"]}}},
            {"$sort": {"__order": 1}}
        ]
        return query

    def get_bow_for_behavior_feature(self, doc):
        try:
            self.add_to_dict(curr_doc=doc["files"], feature_pool=self.files_pool)
            self.add_to_dict(curr_doc=doc["keys"], feature_pool=self.reg_keys_pool)
            self.add_to_dict(curr_doc=doc["mutexes"], feature_pool=self.mutex_pool)
            self.add_to_dict(curr_doc=doc["executed_commands"], feature_pool=self.exec_commands_pool)
        except Exception as e:
            self.log.error(F"Error : {e}")

    def get_bow_for_network_feature(self, doc):
        bow = list()
        for key, value in doc.items():
            if isinstance(value, dict):
                bow += self.get_bow_for_network_feature(value)
            elif isinstance(value, list):
                bow += [str(s) for s in value if isinstance(s, int)]
            else:
                self.log.error(F"Something strange at this Key :{key} \nValue : {value}")
        return bow

    def get_bow_for_static_feature(self, doc):
        bow = list()
        for key, value in doc.items():
            if isinstance(value, list):
                bow += value
            if isinstance(value, dict):
                self.log.info(F"Something strange at this Key :{key} \nValue : {value}")
        return bow

    def get_bow_for_stat_sign_feature(self, doc):
        try:
            bow = [_ for _ in doc]
            return bow
        except Exception as e:
            self.log.error(F"Error : {e}")

    def get_bow_for_docs(self, doc, feature):
        try:
            if feature == "behavior":
                self.get_bow_for_behavior_feature(doc=doc["behavior"])
            if feature == "network":
                network_bow = self.get_bow_for_network_feature(doc=doc["network"])
                self.add_to_dict(curr_doc=network_bow, feature_pool=self.network_pool)
            if feature == "static":
                static_feature_bow = self.get_bow_for_static_feature(doc=doc["static"])
                self.add_to_dict(curr_doc=static_feature_bow, feature_pool=self.static_feature_pool)
            if feature == "statSignatures":
                stat_signatures = self.get_bow_for_stat_sign_feature(doc=doc["statSignatures"])
                self.add_to_dict(curr_doc=stat_signatures, feature_pool=self.stat_sign_pool)
        except Exception as e:
            self.log.error(F"Error : {e}")

    def process_docs(self, c2db_collection, list_of_keys, chunk_size):
        counter = 0

        while counter < len(list_of_keys):
            self.log.info(F"Working on Iter : #{counter / chunk_size}")
            if counter + chunk_size < len(list_of_keys):
                p_keys = list_of_keys[counter: counter + chunk_size]
            else:
                p_keys = list_of_keys[counter:]

            behavior_cursor = c2db_collection.aggregate(self.get_query(list_of_keys=p_keys, feature="behavior"))
            network_cursor = c2db_collection.aggregate(self.get_query(list_of_keys=p_keys, feature="network"))
            static_feature_cursor = c2db_collection.aggregate(self.get_query(list_of_keys=p_keys, feature="static"))
            stat_sign_feature_cursor = c2db_collection.aggregate(
                self.get_query(list_of_keys=p_keys, feature="statSignatures"))

            for doc in behavior_cursor:
                key = doc["key"]
                doc = doc["value"][key]
                self.get_bow_for_docs(doc, feature="behavior")

            for doc in network_cursor:
                key = doc["key"]
                doc = doc["value"][key]
                self.get_bow_for_docs(doc, feature="network")

            for doc in static_feature_cursor:
                key = doc["key"]
                doc = doc["value"][key]
                self.get_bow_for_docs(doc, feature="static")

            for doc in stat_sign_feature_cursor:
                key = doc["key"]
                doc = doc["value"][key]
                self.get_bow_for_docs(doc, feature="statSignatures")

            counter += chunk_size

    def save_feature_pools(self, individual_feature_pool_path, feature_list):
        try:
            self.log.info("Saving files feature cluster")
            json.dump(feature_list[0], open(individual_feature_pool_path + "/" + "files.json", "w"))

            self.log.info("Saving registry keys feature cluster")
            json.dump(feature_list[1], open(individual_feature_pool_path + "/" + "reg_keys.json", "w"))

            self.log.info("Saving mutexes feature cluster")
            json.dump(feature_list[2], open(individual_feature_pool_path + "/" + "mutexes.json", "w"))

            self.log.info("Saving executed commands feature cluster")
            json.dump(feature_list[3], open(individual_feature_pool_path + "/" + "executed_commands.json", "w"))

            self.log.info("Saving network feature cluster")
            json.dump(feature_list[4], open(individual_feature_pool_path + "/" + "network.json", "w"))

            self.log.info("Saving static feature cluster")
            json.dump(feature_list[5], open(individual_feature_pool_path + "/" + "static_features.json", "w"))

            self.log.info("Saving stat signature feature cluster")
            json.dump(feature_list[6], open(individual_feature_pool_path + "/" + "stat_sign_features.json", "w"))
        except Exception as e:
            self.log.error(F"Error : {e}")

    def prune_features(self, top_k_features):
        """
        From the total features, select the top "k" features.
        :param top_k_features: Number of features to select.
        :return:
        """
        files_keys, files_values = self.files_pool.keys(), np.asarray(self.files_pool.values())
        arr = files_values.argsort()[-top_k_features:]
        files_features = [files_keys[_] for _ in arr]

        reg_keys_pool_keys, reg_keys_pool_values = self.reg_keys_pool.keys(), np.asarray(self.reg_keys_pool.values())
        arr = reg_keys_pool_values.argsort()[-top_k_features:]
        reg_keys_pool_features = [reg_keys_pool_keys[_] for _ in arr]

        mutex_keys, mutex_values = self.mutex_pool.keys(), np.asarray(self.mutex_pool.values())
        arr = mutex_values.argsort()[-top_k_features:]
        mutex_features = [mutex_keys[_] for _ in arr]

        exec_commands_keys, exec_commands_values = self.exec_commands_pool.keys(), np.asarray(
            self.exec_commands_pool.values())
        arr = exec_commands_values.argsort()[-top_k_features:]
        exec_commands_features = [exec_commands_keys[_] for _ in arr]

        network_keys, network_values = self.network_pool.keys(), np.asarray(self.network_pool.values())
        arr = network_values.argsort()[-top_k_features:]
        network_features = [network_keys[_] for _ in arr]

        static_keys, static_values = self.static_feature_pool.keys(), np.asarray(self.static_feature_pool.values())
        arr = static_values.argsort()[-top_k_features:]
        static_features = [static_keys[_] for _ in arr]

        stat_sign_keys, stat_sign_values = self.stat_sign_pool.keys(), np.asarray(self.stat_sign_pool.values())
        arr = stat_sign_values.argsort()[-top_k_features:]
        stat_sign_features = [stat_sign_keys[_] for _ in arr]

        return [files_features, reg_keys_pool_features, mutex_features, exec_commands_features,
                network_features,
                static_features,
                stat_sign_features]

    def main(self):
        start_time = time()
        freq_individual_feature_pool_path = self.config["data"]["freq_individual_feature_pool_path"]
        top_k_features = self.config["data"]["top_k_features"]

        self.log.info(F"Preparing Frequency based individual Feature Pools at : {freq_individual_feature_pool_path}")

        self.files_pool = self.get_cluster_dict()
        self.reg_keys_pool = self.get_cluster_dict()
        self.mutex_pool = self.get_cluster_dict()
        self.exec_commands_pool = self.get_cluster_dict()
        self.network_pool = self.get_cluster_dict()
        self.static_feature_pool = self.get_cluster_dict()
        self.stat_sign_pool = self.get_cluster_dict()

        c2db_collection = self.get_collection()
        # list_of_keys = self.get_list_of_keys(c2db_collection=c2db_collection)
        l1 = pi.load(open("/home/satwik/Documents/MachineLearning/Data346k/list_of_keys.pkl"))
        l2 = pi.load(open("/home/satwik/Documents/MachineLearning/Data99k/list_of_keys.pkl"))

        list_of_keys = l1 + l2

        self.process_docs(c2db_collection=c2db_collection, list_of_keys=list_of_keys, chunk_size=100)
        feature_list = self.prune_features(top_k_features=top_k_features)
        self.save_feature_pools(freq_individual_feature_pool_path, feature_list)

        self.log.info(F"Total time taken : {time() - start_time}")


if __name__ == '__main__':
    freq_indi = FreqBasedIndiFeatClusterGen()
    freq_indi.main()
