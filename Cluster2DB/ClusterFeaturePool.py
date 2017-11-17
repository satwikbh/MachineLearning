import json
import urllib
import sys

from collections import defaultdict

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

        client = self.db_utils.get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=is_auth_enabled,
                                          username=username, password=password)

        db_name = self.config['environment']['mongo']['db_name']
        cuckoo_db = client[db_name]

        clusters_db_name = self.config['environment']['mongo']['clusters_db_name']
        clusters_db = client[clusters_db_name]

        c2db_collection_name = self.config['environment']['mongo']['c2db_collection_name']
        c2db_collection = cuckoo_db[c2db_collection_name]

        if family_name not in clusters_db.collection_names():
            clusters_db.create_collection(family_name)
        family_collection = clusters_db[family_name]
        return client, c2db_collection, family_collection

    def get_families_data(self, collection, list_of_keys, config_param_chunk_size):
        classified_families = defaultdict(list)
        unclassified_families = defaultdict(list)

        count = 0
        iteration = 0
        while count < len(list_of_keys):
            try:
                self.log.info("Iteration : {}".format(iteration))
                if count + config_param_chunk_size < len(list_of_keys):
                    p_value = list_of_keys[count:count + config_param_chunk_size]
                else:
                    p_value = list_of_keys[count:]
                n_value = [key.split("_")[1] for key in p_value if "VirusShare_" in key]
                count += config_param_chunk_size
                local_cursor = collection.find({"md5": {"$in": n_value}})
                for index, each_value in enumerate(local_cursor):
                    family = each_value['avclass']['result']
                    val = "VirusShare_" + each_value['md5']
                    if "SINGLETON" in family:
                        unclassified_families[family].append(val)
                    else:
                        classified_families[family].append(val)
                iteration += 1
            except Exception as e:
                self.log.error("Error : {}".format(e))

        malware_families_path = self.config['data']['malware_families_list']

        json.dump(classified_families, open(malware_families_path + "/" + "classified_families.json", "w"))
        json.dump(unclassified_families, open(malware_families_path + "/" + "unclassified_families.json", "w"))

        self.log.info("Classified : {} \t Unclassified : {}".format(len(classified_families),
                                                                    len(unclassified_families)))
        l_value = classified_families.values()
        l_value = self.helper.is_nested_list(l_value)
        return l_value

    def split_into_sub_lists(self, bulk_list, avclass_collection, family_name, iteration):
        chunk_size = 10000
        count, local_iter = 0, iteration
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
        return success, failure, local_iter

    def generate_feature_pool(self, c2db_collection, avclass_collection, list_of_keys, config_param_chunk_size,
                              family_name):
        count, iteration = 0, 0
        success, failure = 0, 0
        while count < len(list_of_keys):
            try:
                self.log.info("Iteration : {}".format(iteration))
                if count + config_param_chunk_size < len(list_of_keys):
                    value = list_of_keys[count:count + config_param_chunk_size]
                else:
                    value = list_of_keys[count:]
                count += config_param_chunk_size
                doc2bow = self.parser.parse_each_document(value, c2db_collection)
                values = self.helper.flatten_list(doc2bow.values())
                size = sys.getsizeof(values) * 1.0 / 10 ** 6
                self.log.info("Number of docs : {}\tSize of docs in MB : {}".format(len(values), size))
                p_success, p_failure, iteration = self.split_into_sub_lists(bulk_list=values,
                                                                            avclass_collection=avclass_collection,
                                                                            family_name=family_name,
                                                                            iteration=iteration)
                success += p_success + 1
                failure += p_failure
            except Exception as e:
                failure += 1
                self.log.error("Error : {}".format(e))
        return success, failure

    def main(self, list_of_keys, family_name):
        client, c2db_collection, family_collection = self.get_collection(family_name)
        config_param_chunk_size = self.config["data"]["config_param_chunk_size"]
        self.log.info("Total keys before AVClass : {}".format(len(list_of_keys)))
        success, failure = self.generate_feature_pool(c2db_collection, family_collection, list_of_keys,
                                                      config_param_chunk_size, family_name)
        self.log.info("Success : {}\tFailure : {}".format(success, failure))
        client.close()


if __name__ == "__main__":
    cfp = ClusterFeaturePool()
    cfp.main(list_of_keys=[], family_name="name")
