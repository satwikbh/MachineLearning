import urllib
import sys

from sshtunnel import SSHTunnelForwarder
from pymongo import MongoClient, InsertOne
from collections import defaultdict
from time import time

from Clustering.KMeansImpl import KMeansImpl
from HelperFunctions.DataStats import DataStats
from HelperFunctions.DistributePoolingSet import DistributePoolingSet
from HelperFunctions.HelperFunction import HelperFunction
from PrepareData.ParsingLogic import ParsingLogic
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from Utils.LoggerUtil import LoggerUtil


class ClusterFeaturePool:
    """
    This class generates the collections in the clusters database.
    Aggregates all the features of the cluster and then splits them into even sized chunks.
    Each chunk is inserted as a document.
    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.db_utils = DBUtils()
        self.parser = ParsingLogic()
        self.dis_pool = DistributePoolingSet()
        self.kmeans = KMeansImpl()
        self.helper = HelperFunction()
        self.config = ConfigUtil().get_config_instance()
        self.data_stats = DataStats()

    def get_collection(self, family_name):
        username = self.config['environment']['mongo']['username']
        pwd = self.config['environment']['mongo']['password']
        password = urllib.quote(pwd)
        address = self.config['environment']['mongo']['address']
        port = self.config['environment']['mongo']['port']
        auth_db = self.config['environment']['mongo']['auth_db']
        is_auth_enabled = self.config['environment']['mongo']['is_auth_enabled']

        local_client = self.db_utils.get_client(address=address, port=port, auth_db=auth_db,
                                                is_auth_enabled=is_auth_enabled,
                                                username=username, password=password)

        db_name = self.config['environment']['mongo']['db_name']
        cuckoo_db = local_client[db_name]

        clusters_db_name = self.config['environment']['mongo']['clusters_db_name']
        clusters_db = local_client[clusters_db_name]

        c2db_collection_name = self.config['environment']['mongo']['c2db_collection_name']
        c2db_collection = cuckoo_db[c2db_collection_name]

        avclass_collection_name = self.config['environment']['mongo']['avclass_collection_name']
        avclass_collection = cuckoo_db[avclass_collection_name]

        if family_name not in clusters_db.collection_names():
            clusters_db.create_collection(family_name)

        family_collection = clusters_db[family_name]

        return local_client, c2db_collection, family_collection, avclass_collection

    def generate_feature_pool(self, c2db_collection, avclass_collection, list_of_keys, config_param_chunk_size,
                              family_name):
        """
        Retrieves all the features of the malware's with the keys and aggregates them.
        Splits them into equal sized chunks which are inserted as documents in the collection.
        The name of the collection matches the family name.
        :param c2db_collection:
        :param avclass_collection:
        :param list_of_keys:
        :param config_param_chunk_size:
        :param family_name:
        :return:
        """
        count, iteration = 0, 0
        values = list()
        while count < len(list_of_keys):
            try:
                self.log.info("Iteration : {}".format(iteration))
                bulk_request_list = list()
                if count + config_param_chunk_size < len(list_of_keys):
                    value = list_of_keys[count:count + config_param_chunk_size]
                else:
                    value = list_of_keys[count:]
                count += config_param_chunk_size
                doc2bow = self.parser.parse_each_document(value, c2db_collection)
                for d_key, d_value in doc2bow.items():
                    value = d_key.split("_")[1]
                    doc = dict()
                    doc["md5"] = value
                    doc["feature_pool"] = d_value
                    doc["malware_source"] = d_key
                    bulk_request_list.append(InsertOne(doc))
                iteration += 1
                try:
                    avclass_collection.bulk_write(bulk_request_list)
                except Exception as e:
                    self.log.error("Bulk Write Error : {}".format(e))
            except Exception as e:
                self.log.error("Error : {}".format(e))

    def get_keys_for_collection(self, family_name, avclass_collection):
        """
        Takes the family_name and searches it in the avclass_collection.
        Returns the list of md5's which belong to the family in the VirusShare_ format.
        :param family_name:
        :param avclass_collection:
        :return:
        """
        family_keys_list = defaultdict(list)
        cursor = avclass_collection.find({"avclass.result": family_name}, {"avclass.result": 1, "md5": 1})
        for doc in cursor:
            try:
                family = doc["avclass"]["result"]
                md5 = doc["md5"]
                family_keys_list[family].append(md5)
            except Exception as e:
                self.log.error("Error : {}".format(e))
        list_of_keys = self.helper.convert_to_vs_keys(family_keys_list[family_name])
        return list_of_keys

    def main(self, family_name):
        start_time = time()
        config_param_chunk_size = self.config["data"]["config_param_chunk_size"]
        local_client, c2db_collection, family_collection, avclass_collection = self.get_collection(family_name)
        list_of_keys = self.get_keys_for_collection(family_name, avclass_collection)
        self.log.info("Total keys : {}".format(len(list_of_keys)))
        self.generate_feature_pool(c2db_collection, family_collection, list_of_keys, config_param_chunk_size,
                                   family_name)
        local_client.close()
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == "__main__":
    cfp = ClusterFeaturePool()
    cfp.main(family_name="")