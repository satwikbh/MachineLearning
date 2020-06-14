import json
from collections import defaultdict
from time import time
from urllib.parse import quote

from HelperFunctions.HelperFunction import HelperFunction
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from Utils.LoggerUtil import LoggerUtil


class IndividualFeatureClusterGeneration:
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
        cluster_dict = defaultdict()
        cluster_dict.default_factory = cluster_dict.__len__
        return cluster_dict

    @staticmethod
    def add_to_dict(curr_doc, feature_pool):
        for _ in curr_doc:
            feature_pool[_]

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
        x = 0
        while x < len(list_of_keys):
            self.log.info(F"Working on Iter : #{x / chunk_size}")
            if x + chunk_size < len(list_of_keys):
                p_keys = list_of_keys[x: x + chunk_size]
            else:
                p_keys = list_of_keys[x:]

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

            x += chunk_size

    def save_feature_pools(self, individual_feature_pool_path):
        try:
            self.log.info("Saving files feature cluster")
            json.dump(self.files_pool, open(individual_feature_pool_path + "/" + "files.json", "w"))

            self.log.info("Saving registry keys feature cluster")
            json.dump(self.reg_keys_pool, open(individual_feature_pool_path + "/" + "reg_keys.json", "w"))

            self.log.info("Saving mutexes feature cluster")
            json.dump(self.mutex_pool, open(individual_feature_pool_path + "/" + "mutexes.json", "w"))

            self.log.info("Saving executed commands feature cluster")
            json.dump(self.exec_commands_pool, open(individual_feature_pool_path + "/" + "executed_commands.json", "w"))

            self.log.info("Saving network feature cluster")
            json.dump(self.network_pool, open(individual_feature_pool_path + "/" + "network.json", "w"))

            self.log.info("Saving static feature cluster")
            json.dump(self.static_feature_pool, open(individual_feature_pool_path + "/" + "static_features.json", "w"))

            self.log.info("Saving stat signature feature cluster")
            json.dump(self.stat_sign_pool, open(individual_feature_pool_path + "/" + "stat_sign_features.json", "w"))
        except Exception as e:
            self.log.error(F"Error : {e}")

    def main(self):
        start_time = time()
        individual_feature_pool_path = self.config["data"]["individual_feature_pool_path"]
        self.log.info(F"Preparing Individual Feature Pools at : {individual_feature_pool_path}")

        self.files_pool = self.get_cluster_dict()
        self.reg_keys_pool = self.get_cluster_dict()
        self.mutex_pool = self.get_cluster_dict()
        self.exec_commands_pool = self.get_cluster_dict()
        self.network_pool = self.get_cluster_dict()
        self.static_feature_pool = self.get_cluster_dict()
        self.stat_sign_pool = self.get_cluster_dict()

        c2db_collection = self.get_collection()
        list_of_keys = self.get_list_of_keys(c2db_collection=c2db_collection)
        self.process_docs(c2db_collection=c2db_collection, list_of_keys=list_of_keys, chunk_size=500)
        self.save_feature_pools(individual_feature_pool_path)

        self.log.info(F"Total time taken : {time() - start_time}")


if __name__ == '__main__':
    indi_fc = IndividualFeatureClusterGeneration()
    indi_fc.main()
