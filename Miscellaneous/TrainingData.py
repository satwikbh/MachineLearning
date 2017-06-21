import json
from collections import defaultdict

from pymongo import MongoClient

from Utils.LoggerUtil import LoggerUtil


class TrainingData:
    def __init__(self):
        self.log = LoggerUtil(self.__class__).get()
        self.final_result = list()
        self.clusters = defaultdict(list)

    def get_train_data(self):
        """
        This function will return all the malware's which are classified by Malheur and given a score of 10.0
        :return:
        """
        # client = MongoClient("mongodb://" + "admin" + ":" + urllib.quote("goodDevelopers@123") + "@" + "localhost:27017" + "/" + "admin")
        client = MongoClient("mongodb://" + "localhost:27017" + "/" + "admin")
        db = client['cuckoo']
        collection = db['cluster2db']

        list_of_docs = collection.distinct("key")

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
            if each['score'] >= 10.0:
                self.clusters[each['family']].append(each)

        return json.dumps(self.clusters)


if __name__ == '__main__':
    train_data = TrainingData()
    print train_data.get_train_data()
