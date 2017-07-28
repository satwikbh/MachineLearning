import time
import hickle
import numpy as np

from collections import defaultdict
from scipy.sparse import coo_matrix, vstack

from Utils.LoggerUtil import LoggerUtil
from DistributePoolingSet import DistributePoolingSet
from HelperFunction import HelperFunction


class ParsingLogic:
    def __init__(self):
        self.log = LoggerUtil(self.__class__).get()
        self.dis_pool = DistributePoolingSet()
        self.helper = HelperFunction()

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
        start_time = time.time()
        cursor = collection.find({"key": {"$in": list_of_docs}})

        for each_document in cursor:
            feature = each_document.get("feature")
            value = each_document.get("value")
            if feature == "behavior" or feature == "network" or feature == "static" or feature == "statSignatures":
                list_of_keys = value.values()[0].keys()
                if feature in list_of_keys:
                    d2b = self.get_bow_for_each_document(value, feature)
                    if d2b is not None:
                        doc2bow[each_document.get("key")] += d2b

        self.log.info("Time taken for Parsing the documents : {}".format(time.time() - start_time))
        return doc2bow

    def convert2vec(self, feature_pool_part_path_list, feature_vector_path, num_rows):
        """
        Generate & return the feature vector path names
        The feature matrix is in Scipy CSR format.
        :return: 
        """
        self.log.info("************ Convert 2 Vector *************")
        start_time = time.time()
        feature_vector_list = list()
        fv_dist_fnames = list()
        cluster_dict = dict()

        for index, each_file in enumerate(feature_pool_part_path_list):
            file_object = open(each_file)
            doc2bow = hickle.load(file_object)
            doc2bow_list = doc2bow.flatten()
            for each in doc2bow_list:
                cluster_dict.setdefault(each, len(cluster_dict))
            del doc2bow, doc2bow_list
            file_object.close()

        num_cols = len(cluster_dict.keys())
        self.log.info("Input Matrix Shape : (Rows={}, Columns={})".format(num_rows, num_cols))

        for index, each_file in enumerate(feature_pool_part_path_list):
            file_object = open(each_file)
            doc2bow = hickle.load(file_object)
            matrix = list()
            for inner_index, each in enumerate(doc2bow):
                column = [cluster_dict.get(x) for x in each]
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

        self.log.info("Time taken for Convert 2 Vector : {}".format(time.time() - start_time))
        return fv_dist_fnames
