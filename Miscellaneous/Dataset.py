import json
import urllib
from collections import defaultdict

from pymongo import MongoClient

from Utils.LoggerUtil import LoggerUtil


class Dataset:
    def __init__(self):
        self.log = LoggerUtil(self.__class__).get()
        self.final_result = list()

    def get_train_data(self, list_of_docs, collection):
        """
        This function will return all the malware's which are classified by Malheur and given a score of 10.0
        :return:
        """
        clusters = defaultdict(list)
        for index, each in enumerate(list_of_docs):
            try:
                query = {"key": each, "feature": "malheur"}
                cursor = collection.find(query)
                result = cursor[0]['value'][each]['malheur']
                temp = dict()
                temp['malware'] = each
                temp['score'] = result['score']
                temp['family'] = result['family']
                self.final_result.append(temp)
            except Exception as e:
                self.log.error("Error caused by :", index, query, each, e)

        for each in self.final_result:
            if each['score'] == 10.0:
                clusters[each['family']].append(each)

        return json.dumps(clusters)

    def get_test_data(self, list_of_docs, collection):
        clusters = defaultdict(list)
        for index, each in enumerate(list_of_docs):
            try:
                query = {"key": each, "feature": "malheur"}
                cursor = collection.find(query)
                result = cursor[0]['value'][each]['malheur']
                temp = dict()
                temp['malware'] = each
                temp['score'] = result['score']
                temp['family'] = result['family']
                self.final_result.append(temp)
            except Exception as e:
                self.log.error("Error caused by :", index, query, each, e)

        for each in self.final_result:
            if each['score'] == 0.0:
                clusters[each['family']].append(each)

        return json.dumps(clusters)

    def get_data(self, test=None, train=None):
        client = MongoClient(
            "mongodb://" + "admin" + ":" + urllib.quote("goodDevelopers@123") + "@" + "localhost:27017" + "/" + "admin")
        # client = MongoClient("mongodb://" + "localhost:27017" + "/" + "admin")
        db = client['cuckoo']
        collection = db['cluster2db']

        list_of_docs = collection.distinct("key")
        if train or test:
            if train:
                train_data = self.get_train_data(list_of_docs, collection)
                json.dump(train_data, open("train_data.json", 'w'))
            if test:
                test_data = self.get_test_data(list_of_docs, collection)
                json.dump(test_data, open("test_data.json", 'w'))
        else:
            self.log.error("Error. \nWhich dataset to generate not specified")


if __name__ == '__main__':
    data = Dataset()
    print data.get_data(train=True, test=True)
