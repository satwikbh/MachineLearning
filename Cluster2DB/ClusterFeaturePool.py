import urllib
import sys

from sshtunnel import SSHTunnelForwarder
from pymongo import MongoClient

from Clustering.KMeansImpl import KMeansImpl
from HelperFunctions.DataStats import DataStats
from HelperFunctions.DistributePoolingSet import DistributePoolingSet
from HelperFunctions.HelperFunction import HelperFunction
from PrepareData.ParsingLogic import ParsingLogic
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from Utils.LoggerUtil import LoggerUtil


class ClusterFeaturePool:
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

        server = SSHTunnelForwarder("10.2.40.13", ssh_username="satwik", ssh_password="aith0561",
                                    remote_bind_address=('127.0.0.1', 27017))
        server.start()
        remote_client = MongoClient('127.0.0.1', server.local_bind_port)

        db_name = self.config['environment']['mongo']['db_name']
        cuckoo_db = local_client[db_name]

        clusters_db_name = self.config['environment']['mongo']['clusters_db_name']
        clusters_db = remote_client[clusters_db_name]

        c2db_collection_name = self.config['environment']['mongo']['c2db_collection_name']
        c2db_collection = cuckoo_db[c2db_collection_name]

        if family_name not in clusters_db.collection_names():
            clusters_db.create_collection(family_name)
        family_collection = clusters_db[family_name]
        return local_client, c2db_collection, family_collection

    def split_into_sub_lists(self, bulk_list, avclass_collection, family_name):
        chunk_size = 10000
        count, local_iter = 0, 0
        success, failure = 0, 0
        while count < len(bulk_list):
            try:
                avclass_collection.insert_one({'family_name': family_name, 'feature_pool': [], "index": local_iter})
                if count + chunk_size < len(bulk_list):
                    values = bulk_list[count:count + chunk_size]
                else:
                    values = bulk_list[count:]
                avclass_collection.update_one({'family_name': family_name, 'index': local_iter},
                                              {'$push': {'feature_pool': values}})
                count += chunk_size
                success += 1
                local_iter += 1
            except Exception as e:
                failure += 1
                self.log.error("Error : {}".format(e))
        self.log.info("Success : {}\tFailure : {}".format(success, failure))

    def generate_feature_pool(self, c2db_collection, avclass_collection, list_of_keys, config_param_chunk_size,
                              family_name):
        count, iteration = 0, 0
        values = list()
        while count < len(list_of_keys):
            try:
                self.log.info("Iteration : {}".format(iteration))
                if count + config_param_chunk_size < len(list_of_keys):
                    value = list_of_keys[count:count + config_param_chunk_size]
                else:
                    value = list_of_keys[count:]
                count += config_param_chunk_size
                doc2bow = self.parser.parse_each_document(value, c2db_collection)
                values += self.helper.flatten_list(doc2bow.values())
                size = sys.getsizeof(values) * 1.0 / 10 ** 6
                self.log.info("Number of docs : {}\tSize of docs in MB : {}".format(len(values), size))
                iteration += 1
                value = [x.split("_")[1] for x in value]
                avclass_collection.insert_one({"md5": value})
            except Exception as e:
                self.log.error("Error : {}".format(e))
        values = list(set(values))
        self.split_into_sub_lists(bulk_list=values,
                                  avclass_collection=avclass_collection,
                                  family_name=family_name)

    def main(self, list_of_keys, family_name):
        client, c2db_collection, family_collection = self.get_collection(family_name)
        config_param_chunk_size = self.config["data"]["config_param_chunk_size"]
        self.log.info("Total keys : {}".format(len(list_of_keys)))
        self.generate_feature_pool(c2db_collection, family_collection, list_of_keys, config_param_chunk_size,
                                   family_name)
        client.close()


if __name__ == "__main__":
    cfp = ClusterFeaturePool()
    cfp.main(list_of_keys=[], family_name="name")
