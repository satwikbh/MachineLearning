import time
import json
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

        for index, each_document in enumerate(list_of_docs):
            cursor = collection.find({"key": each_document})
            for each in cursor:
                feature = each.get("feature")
                value = each.get("value")
                if feature == "behavior" or feature == "network" or feature == "static" or feature == "statSignatures":
                    list_of_keys = value.values()[0].keys()
                    if feature in list_of_keys:
                        d2b = self.get_bow_for_each_document(value, feature)
                        if d2b is not None:
                            doc2bow[each.get("key")] += d2b
        self.log.info("Time taken for Parsing the documents : {}".format(time.time() - start_time))
        return doc2bow

    def convert2vec(self, dist_fnames, num_rows):
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

        dist_fnames, counter = self.dis_pool.load_distributed_feature_vector(dist_fnames)

        for index, each_file in enumerate(dist_fnames):
            doc2bow_str = hickle.load(open(each_file))
            doc2bow = json.loads(doc2bow_str)
            flat_set = set()
            for sublist in doc2bow.values():
                for item in sublist:
                    if item not in flat_set:
                        cluster_dict.setdefault(item, len(cluster_dict))

        num_cols = len(cluster_dict.keys())
        self.log.info("Input Matrix Shape : (Rows={}, Columns={})".format(num_rows, num_cols))

        for index, each_file in enumerate(dist_fnames):
            doc2bow_str = hickle.load(open(each_file))
            doc2bow = json.loads(doc2bow_str)
            matrix = list()
            for inner_index, each in enumerate(doc2bow.values()):
                column = [cluster_dict.get(x) for x in each]
                row = len(column) * [0]
                data = len(column) * [1.0]
                value = coo_matrix((data, (row, column)), shape=(1, len(cluster_dict.keys())), dtype=np.float32)
                matrix.append(value)
                self.log.info("Working on {} matrix and {} sub-matrix".format(index, inner_index))

            mini_batch_matrix = vstack(matrix)
            fv_dist_fnames.append(self.dis_pool.save_distributed_feature_vector(mini_batch_matrix, index + counter))
            feature_vector_list.append(mini_batch_matrix)

        self.log.info("Time taken for Convert 2 Vector : {}".format(time.time() - start_time))
        return fv_dist_fnames, len(cluster_dict.keys())
