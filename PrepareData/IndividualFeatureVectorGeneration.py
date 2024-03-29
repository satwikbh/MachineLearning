import glob
import pickle as pi
import json
from time import time
from urllib.parse import quote

import numpy as np
from scipy.sparse import coo_matrix, vstack

from HelperFunctions.DistributePoolingSet import DistributePoolingSet
from HelperFunctions.HelperFunction import HelperFunction
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from Utils.LoggerUtil import LoggerUtil


class IndividualFeatureGeneration:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.db_utils = DBUtils()
        self.config = ConfigUtil.get_config_instance()
        self.dis_pool = DistributePoolingSet()
        self.files_pool = None
        self.reg_keys_pool = None
        self.mutex_pool = None
        self.exec_commands_pool = None
        self.network_pool = None
        self.static_feature_pool = None
        self.stat_sign_feature_pool = None

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
        cursor = c2db_collection.aggregate([{"$group": {"_id": "$key"}}], allowDiskUse=True)
        list_of_keys = self.helper.cursor_to_list(cursor, identifier="_id")
        return list_of_keys

    def load_feature_pools(self, indi_feature_path):
        list_of_files = glob.glob(indi_feature_path + "/" + "*.json")
        for file_path in list_of_files:
            if "files" in file_path:
                self.files_pool = json.load(open(file_path))
            elif "reg_keys" in file_path:
                self.reg_keys_pool = json.load(open(file_path))
            elif "mutexes" in file_path:
                self.mutex_pool = json.load(open(file_path))
            elif "executed_commands" in file_path:
                self.exec_commands_pool = json.load(open(file_path))
            elif "network" in file_path:
                self.network_pool = json.load(open(file_path))
            elif "static_features" in file_path:
                self.static_feature_pool = json.load(open(file_path))
            elif "stat_sign_features" in file_path:
                self.stat_sign_feature_pool = json.load(open(file_path))
            else:
                self.log.error("Something not in feature list accessed")

    def process_docs(self, **kwargs):
        chunk_size = kwargs['chunk_size']
        c2db_collection = kwargs['c2db_collection']
        list_of_keys = kwargs['list_of_keys']
        files_fv_path = kwargs['files_fv_path']
        reg_keys_fv_path = kwargs['reg_keys_fv_path']
        mutexes_fv_path = kwargs['mutexes_fv_path']
        exec_cmds_fv_path = kwargs['exec_cmds_fv_path']
        network_fv_path = kwargs['network_fv_path']
        static_feature_fv_path = kwargs['static_feature_fv_path']

        list_of_files_feature = list()
        list_of_reg_keys_feature = list()
        list_of_mutex_features = list()
        list_of_exec_commands_features = list()
        list_of_network_features = list()
        list_of_static_features = list()

        x = 0
        index = 0

        while x < len(list_of_keys):
            self.log.info(F"Working on Iter : #{x / chunk_size}")
            if x + chunk_size < len(list_of_keys):
                p_keys = list_of_keys[x: x + chunk_size]
            else:
                p_keys = list_of_keys[x:]

            behavior_cursor = c2db_collection.aggregate(self.get_query(list_of_keys=p_keys, feature="behavior"), allowDiskUse=True)
            network_cursor = c2db_collection.aggregate(self.get_query(list_of_keys=p_keys, feature="network"), allowDiskUse=True)
            static_feature_cursor = c2db_collection.aggregate(self.get_query(list_of_keys=p_keys, feature="static"), allowDiskUse=True)

            for doc in behavior_cursor:
                key = doc["key"]
                doc = doc["value"][key]
                files_value, reg_keys_value, mutex_value, exec_cmds_value = self.get_bow_for_docs(doc,
                                                                                                  feature="behavior")
                list_of_files_feature.append(files_value)
                list_of_reg_keys_feature.append(reg_keys_value)
                list_of_mutex_features.append(mutex_value)
                list_of_exec_commands_features.append(exec_cmds_value)

            for doc in network_cursor:
                key = doc["key"]
                doc = doc["value"][key]
                network_value = self.get_bow_for_docs(doc, feature="network")
                list_of_network_features.append(network_value)

            for doc in static_feature_cursor:
                key = doc["key"]
                doc = doc["value"][key]
                static_feature_value = self.get_bow_for_docs(doc, feature="static")
                list_of_static_features.append(static_feature_value)

            files_matrix = vstack(list_of_files_feature)
            reg_keys_matrix = vstack(list_of_reg_keys_feature)
            mutex_matrix = vstack(list_of_mutex_features)
            exec_commands_matrix = vstack(list_of_exec_commands_features)
            network_matrix = vstack(list_of_network_features)
            static_features_matrix = vstack(list_of_static_features)

            self.save_individual_feature_vector(files_matrix=files_matrix, reg_keys_matrix=reg_keys_matrix,
                                                mutex_matrix=mutex_matrix, exec_commands_matrix=exec_commands_matrix,
                                                network_matrix=network_matrix,
                                                static_features_matrix=static_features_matrix,
                                                index=index, files_fv_path=files_fv_path,
                                                reg_keys_fv_path=reg_keys_fv_path,
                                                mutexes_fv_path=mutexes_fv_path, exec_cmds_fv_path=exec_cmds_fv_path,
                                                network_fv_path=network_fv_path,
                                                static_feature_fv_path=static_feature_fv_path)
            index += 1
            x += chunk_size

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

    def get_bow_for_docs(self, doc, feature):
        if feature == "behavior":
            files_value, reg_keys_value, mutex_value, exec_cmds_value = self.get_bow_for_behavior_feature(
                doc=doc["behavior"])
            return files_value, reg_keys_value, mutex_value, exec_cmds_value
        if feature == "network":
            network_value = self.get_bow_for_network_feature(doc=doc["network"])
            return network_value
        if feature == "static":
            static_feature_value = self.get_bow_for_static_feature(doc=doc["static"])
            return static_feature_value

    def get_bow_for_behavior_feature(self, doc):
        files_value = self.gen_vector(feature_pool=self.files_pool, doc_feature=doc["files"])
        reg_keys_value = self.gen_vector(feature_pool=self.reg_keys_pool, doc_feature=doc["keys"])
        mutex_value = self.gen_vector(feature_pool=self.mutex_pool, doc_feature=doc["mutexes"])
        exec_commands_value = self.gen_vector(feature_pool=self.exec_commands_pool,
                                              doc_feature=doc["executed_commands"])

        return files_value, reg_keys_value, mutex_value, exec_commands_value

    def parse_network_feature(self, feature, doc):
        bow = list()
        for key, value in doc.items():
            if isinstance(value, dict):
                bow += self.parse_network_feature(feature, value)
            elif isinstance(value, list):
                bow += [str(s) for s in value if isinstance(s, int)]
            else:
                self.log.error(F"In feature {feature} \nSomething strange at this Key :{key} \nValue : {value}")
        return bow

    def get_bow_for_network_feature(self, doc):
        doc = self.parse_network_feature(feature="network", doc=doc)
        network_value = self.gen_vector(feature_pool=self.network_pool, doc_feature=doc)
        return network_value

    def parse_static_feature(self, feature, doc):
        bow = list()
        for key, value in doc.items():
            if isinstance(value, list):
                bow += value
            if isinstance(value, dict):
                self.log.error(F"In feature {feature} \nSomething strange at this Key :{key} \nValue : {value}")
        return bow

    def get_bow_for_static_feature(self, doc):
        doc = self.parse_static_feature(feature="static", doc=doc)
        static_feature_value = self.gen_vector(feature_pool=self.static_feature_pool, doc_feature=doc)
        return static_feature_value

    @staticmethod
    def gen_vector(feature_pool, doc_feature):
        column = [feature_pool.get(x) for x in doc_feature if x in feature_pool]
        row = len(column) * [0]
        data = len(column) * [1.0]
        value = coo_matrix((data, (row, column)), shape=(1, len(feature_pool.keys())), dtype=np.int8)
        return value

    def save_individual_feature_vector(self, **kwargs):
        index = kwargs['index']

        files_matrix = kwargs['files_matrix']
        reg_keys_matrix = kwargs['reg_keys_matrix']
        mutex_matrix = kwargs['mutex_matrix']
        exec_commands_matrix = kwargs['exec_commands_matrix']
        network_matrix = kwargs['network_matrix']
        static_features_matrix = kwargs['static_features_matrix']

        files_fv_path = kwargs['files_fv_path']
        reg_keys_fv_path = kwargs['reg_keys_fv_path']
        mutexes_fv_path = kwargs['mutexes_fv_path']
        exec_cmds_fv_path = kwargs['exec_cmds_fv_path']
        network_fv_path = kwargs['network_fv_path']
        static_feature_fv_path = kwargs['static_feature_fv_path']

        self.dis_pool.save_distributed_feature_vector(files_matrix,
                                                      files_fv_path,
                                                      index)
        self.dis_pool.save_distributed_feature_vector(reg_keys_matrix,
                                                      reg_keys_fv_path,
                                                      index)
        self.dis_pool.save_distributed_feature_vector(mutex_matrix,
                                                      mutexes_fv_path,
                                                      index)
        self.dis_pool.save_distributed_feature_vector(exec_commands_matrix,
                                                      exec_cmds_fv_path,
                                                      index)
        self.dis_pool.save_distributed_feature_vector(network_matrix,
                                                      network_fv_path,
                                                      index)
        self.dis_pool.save_distributed_feature_vector(static_features_matrix,
                                                      static_feature_fv_path,
                                                      index)

    def main(self):
        start_time = time()
        chunk_size = 1000

        individual_feature_pool_path = self.config["data"]["individual_feature_pool_path"]
        files_fv_path = self.config["individual_feature_vector_path"]["files_feature"]
        self.helper.create_dir_if_absent(files_fv_path)
        reg_keys_fv_path = self.config["individual_feature_vector_path"]["reg_keys_feature"]
        self.helper.create_dir_if_absent(reg_keys_fv_path)
        mutexes_fv_path = self.config["individual_feature_vector_path"]["mutexes_feature"]
        self.helper.create_dir_if_absent(mutexes_fv_path)
        exec_cmds_fv_path = self.config["individual_feature_vector_path"]["exec_cmds_feature"]
        self.helper.create_dir_if_absent(exec_cmds_fv_path)
        network_fv_path = self.config["individual_feature_vector_path"]["network_feature"]
        self.helper.create_dir_if_absent(network_fv_path)
        static_feature_fv_path = self.config["individual_feature_vector_path"]["static_feature"]
        self.helper.create_dir_if_absent(static_feature_fv_path)

        c2db_collection = self.get_collection()
        # list_of_keys = self.get_list_of_keys(c2db_collection=c2db_collection)
        list_of_keys = json.load(open("/home/satwik/Documents/MachineLearning/Data346k/list_of_keys.json"))
        self.load_feature_pools(indi_feature_path=individual_feature_pool_path)
        self.process_docs(c2db_collection=c2db_collection, list_of_keys=list_of_keys, chunk_size=chunk_size,
                          files_fv_path=files_fv_path, reg_keys_fv_path=reg_keys_fv_path,
                          mutexes_fv_path=mutexes_fv_path, exec_cmds_fv_path=exec_cmds_fv_path,
                          network_fv_path=network_fv_path, static_feature_fv_path=static_feature_fv_path)

        self.log.info(F"Time taken for Convert 2 Vector : {time() - start_time}")


if __name__ == '__main__':
    indi_fg = IndividualFeatureGeneration()
    indi_fg.main()
