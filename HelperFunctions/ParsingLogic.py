import time
from collections import defaultdict

from Utils.LoggerUtil import LoggerUtil


class ParsingLogic:
    def __init__(self):
        self.doc2bow = defaultdict(list)
        self.log = LoggerUtil(self.__class__).get()

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
        self.log.info("************ Parsing the documents *************")
        start_time = time.time()
        for each_document in list_of_docs:
            cursor = collection.find({"key": each_document})
            for each in cursor:
                feature = each.get("feature")
                value = each.get("value")
                if feature == "behavior" or feature == "network" or feature == "static" or feature == "statSignatures":
                    list_of_keys = value.values()[0].keys()
                    if feature in list_of_keys:
                        d2b = self.get_bow_for_each_document(value, feature)
                        if d2b is not None:
                            self.doc2bow[each.get("key")] += d2b
        self.log.info("Time taken for Parsing the documents : {}".format(time.time() - start_time))

    def convert2vec(self):
        """
        Generate & return the feature vector.
        :return: 
        """
        self.log.info("************ Convert 2 Vector *************")
        start_time = time.time()
        flat_list = [item for sublist in self.doc2bow.values() for item in sublist]
        cluster = list(set(flat_list))
        feature_vector = list()

        for index, each in enumerate(self.doc2bow.values()):
            temp = len(cluster) * ['0']
            sparse_representation = [cluster.index(x) for x in each]
            for x in sparse_representation:
                temp[x] = '1'
            value = ''.join(temp)
            del temp
            feature_vector.append(value)

        self.log.info("Time taken for Convert 2 Vector : {}".format(time.time() - start_time))
        return feature_vector
