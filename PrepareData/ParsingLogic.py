import glob
import json
import pickle as pi
from collections import defaultdict
from time import time

import numpy as np
from scipy.sparse import coo_matrix, vstack

from HelperFunctions.DistributePoolingSet import DistributePoolingSet
from HelperFunctions.HelperFunction import HelperFunction
from Utils.LoggerUtil import LoggerUtil


class ParsingLogic:
    def __init__(self, use_trie_pruning):
        self.log = LoggerUtil(self.__class__).get()
        self.dis_pool = DistributePoolingSet()
        self.helper = HelperFunction()
        self.use_trie_pruning = use_trie_pruning
        self.files_pool = None
        self.reg_keys_pool = None
        self.mutex_pool = None
        self.exec_commands_pool = None
        self.network_pool = None
        self.static_feature_pool = None
        self.stat_sign_feature_pool = None

    def get_bow_for_behavior_feature(self, feature, doc):
        bow = list()
        for key, value in doc.items():
            if isinstance(value, list):
                bow += value
            else:
                self.log.error(
                    "In feature {} \nSomething strange at this Key :{} \nValue : {}".format(feature, key, value))
        return bow

    def get_bow_for_network_feature(self, feature, doc):
        bow = list()
        for key, value in doc.items():
            if isinstance(value, dict):
                bow += self.get_bow_for_network_feature(feature, value)
            elif isinstance(value, list):
                bow += [str(s) for s in value if isinstance(s, int)]
            else:
                self.log.error(
                    "In feature {} \nSomething strange at this Key :{} \nValue : {}".format(feature, key, value))
        return bow

    def get_bow_for_statistic_feature(self, feature, doc):
        bow = list()
        if isinstance(doc, list):
            bow += doc
        else:
            self.log.error("Feature {} doesn't have {} type as value.".format(feature, type(doc)))

    def get_bow_for_static_feature(self, feature, doc):
        bow = list()
        for key, value in doc.items():
            if isinstance(value, list):
                bow += value
            if isinstance(value, dict):
                self.log.error(
                    "In feature {} \nSomething strange at this Key :{} \nValue : {}".format(feature, key, value))
        return bow

    def get_bow_for_each_document(self, document, feature):
        if feature == "behavior":
            behavior = document.values()[0].get(feature)
            return self.get_bow_for_behavior_feature(feature, behavior)
        elif feature == "network":
            network = document.values()[0].get(feature)
            return self.get_bow_for_network_feature(feature, network)
        elif feature == "static":
            static = document.values()[0].get(feature)
            return self.get_bow_for_static_feature(feature, static)
        elif feature == "statSignatures":
            statistic = document.values()[0].get(feature)
            return self.get_bow_for_statistic_feature(feature, statistic)
        else:
            self.log.error("Feature other than behavior, network, static, statistic accessed.")
            return None

    def parse_each_document(self, list_of_docs, collection):
        doc2bow = defaultdict(list)
        self.log.info("************ Parsing the documents *************")
        start_time = time()
        query = [
            {"$match": {"key": {"$in": list_of_docs}}},
            {"$addFields": {"__order": {"$indexOfArray": [list_of_docs, "$key"]}}},
            {"$sort": {"__order": 1}}
        ]
        cursor = collection.aggregate(query)

        for each_document in cursor:
            feature = each_document.get("feature")
            value = each_document.get("value")
            if feature == "behavior" or feature == "network" or feature == "static" or feature == "statSignatures":
                list_of_keys = value.values()[0].keys()
                if feature in list_of_keys:
                    d2b = self.get_bow_for_each_document(value, feature)
                    if d2b is not None:
                        doc2bow[each_document.get("key")] += d2b

        self.log.info("Time taken for Parsing the documents : {}".format(time() - start_time))
        return doc2bow

    def load_feature_pools(self, indi_feature_pool_path):
        list_of_files = glob.glob(indi_feature_pool_path + "/" + "*.json")
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

    def pruning_feature_cluster(self, pruned_feature_pool_path):
        """
        From the list of features, a trie is constructed and the values are pruned.
        The values from each feature are segregated into their respective feature pools.
        The cluster dict is constructed from these pools.
        :param pruned_feature_pool_path:
        :return:
        """
        self.log.info("************ Convert 2 Vector *************")
        start_time = time()
        cluster_dict = defaultdict(list)
        cluster_dict.default_factory = cluster_dict.__len__

        self.load_feature_pools(indi_feature_pool_path=pruned_feature_pool_path)
        self.log.info("Final pool preparation")
        for feature in self.files_pool:
            cluster_dict[feature]
        for feature in self.reg_keys_pool:
            cluster_dict[feature]
        for feature in self.mutex_pool:
            cluster_dict[feature]
        for feature in self.exec_commands_pool:
            cluster_dict[feature]
        for feature in self.network_pool:
            cluster_dict[feature]
        for feature in self.static_feature_pool:
            cluster_dict[feature]
        for feature in self.stat_sign_feature_pool:
            cluster_dict[feature]

        self.log.info("Time taken for generating final feature pool : {}".format(time() - start_time))
        return cluster_dict

    def non_pruning_feature_cluster(self, feature_pool_part_path_list):
        """
        This uses the non-pruning method to construct the cluster dict.
        :param feature_pool_part_path_list:
        :return:
        """
        self.log.info("************ Convert 2 Vector *************")
        start_time = time()
        cluster_dict = defaultdict(list)
        cluster_dict.default_factory = cluster_dict.__len__

        for index, each_file in enumerate(feature_pool_part_path_list):
            self.log.info("Final pool preparation Iteration: #{}".format(index))
            file_object = open(each_file)
            doc2bow = pi.load(file_object)
            flat_list = self.helper.flatten_list(doc2bow)
            for each in flat_list:
                cluster_dict[each]
            del doc2bow, flat_list
            file_object.close()

        self.log.info("Time taken for generating final feature pool : {}".format(time() - start_time))
        return cluster_dict

    def convert2vec(self, pruned_feature_pool_path, feature_pool_part_path_list, feature_vector_path, num_rows):
        """
        Generate & return the feature vector path names
        The feature matrix is in Scipy CSR format.
        :param pruned_feature_pool_path:
        :param feature_pool_part_path_list:
        :param feature_vector_path:
        :param num_rows:
        :return:
        """
        feature_vector_list = list()
        fv_dist_fnames = list()

        if self.use_trie_pruning:
            cluster_dict = self.pruning_feature_cluster(pruned_feature_pool_path=pruned_feature_pool_path)
        else:
            cluster_dict = self.non_pruning_feature_cluster(feature_pool_part_path_list=feature_pool_part_path_list)
        num_cols = len(cluster_dict.keys())
        self.log.info("Input Matrix Shape : (Rows={}, Columns={})".format(num_rows, num_cols))

        start_time = time()
        for index, each_file in enumerate(feature_pool_part_path_list):
            file_object = open(each_file)
            doc2bow = pi.load(file_object)
            matrix = list()
            for inner_index, each in enumerate(doc2bow):
                column = list(set([cluster_dict.get(x) for x in each]))
                row = len(column) * [0]
                data = len(column) * [1.0]

                value = coo_matrix((data, (row, column)), shape=(1, len(cluster_dict.keys())), dtype=np.float32)
                matrix.append(value)
                self.log.info("Working on {} matrix and {} sub-matrix".format(index, inner_index))

            mini_batch_matrix = vstack(matrix)
            fv_dist_part_file_name = self.dis_pool.save_distributed_feature_vector(mini_batch_matrix,
                                                                                   feature_vector_path,
                                                                                   index)
            fv_dist_fnames.append(fv_dist_part_file_name)
            feature_vector_list.append(mini_batch_matrix)
            file_object.close()

        self.log.info("Time taken for Convert 2 Vector : {}".format(time() - start_time))
        return fv_dist_fnames
