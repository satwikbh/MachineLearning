import urllib
import json

from collections import defaultdict
from time import time

from Utils.LoggerUtil import LoggerUtil
from Utils.DBUtils import DBUtils
from Utils.ConfigUtil import ConfigUtil
from HelperFunctions.HelperFunction import HelperFunction


class AvclassDistribution(object):
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.db_utils = DBUtils()
        self.config = ConfigUtil.get_config_instance()
        self.meta_dict = defaultdict()

    def get_collection(self):
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
        db = client[db_name]

        c2db_collection_name = self.config['environment']['mongo']['c2db_collection_name']
        c2db_collection = db[c2db_collection_name]

        avclass_collection_name = self.config['environment']['mongo']['avclass_collection_name']
        avclass_collection = db[avclass_collection_name]

        return c2db_collection, avclass_collection

    def get_freq_dict(self, document):
        verbose = document["avclass"]["verbose"]
        md5 = document["md5"]
        if len(verbose) == 1:
            family_name, score = verbose[0]
            self.meta_dict[md5] = [{"family_name": family_name, "score": 1}]
        else:
            tmp = list()
            total_score = 0
            for inner_list in verbose:
                family_name, score = inner_list
                total_score += score
            for inner_list in verbose:
                family_name, score = inner_list
                tmp.append({"family_name": family_name, "score": (score * 1.0) / total_score})
            self.meta_dict[md5] = tmp

    def process(self, list_of_keys, collection):
        count = 0
        index = 0
        chunk_size = 1000

        while count < len(list_of_keys):
            self.log.info("Working on Iter : #{}".format(index))
            if count + chunk_size < len(list_of_keys):
                p_keys = list_of_keys[count: count + chunk_size]
            else:
                p_keys = list_of_keys[count:]

            query = [
                {"$match": {"md5": {"$in": p_keys}}},
                {"$project": {"avclass.verbose": 1, "md5": 1}},
                {"$addFields": {"__order": {"$indexOfArray": [p_keys, "$md5"]}}},
                {"$sort": {"__order": 1}}
            ]
            cursor = collection.aggregate(query)
            for doc in cursor:
                self.get_freq_dict(doc)
            count += chunk_size
            index += 1

    def get_list_of_keys(self, collection):
        cursor = collection.aggregate([{"$group": {"_id": '$key'}}])
        list_of_keys = self.helper.cursor_to_list(cursor, identifier="_id")
        return list_of_keys

    def main(self):
        start_time = time()
        keys_path = self.config["data"]["list_of_keys"]
        c2db_collection, avclass_collection = self.get_collection()
        list_of_vs_keys = self.get_list_of_keys(collection=c2db_collection)
        self.log.info("Total Number of keys : {}".format(len(list_of_vs_keys)))
        list_of_keys = self.helper.convert_from_vs_keys(list_of_vs_keys=list_of_vs_keys)
        self.process(list_of_keys=list_of_keys, collection=avclass_collection)
        json.dump(self.meta_dict, open(keys_path + "/" + "avclass_distribution.json", "w"))
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    ranking = AvclassDistribution()
    ranking.main()
