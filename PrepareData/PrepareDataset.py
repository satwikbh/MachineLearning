import json
import math
import pickle as pi
import urllib

from collections import defaultdict

from Clustering.KMeansImpl import KMeansImpl
from HelperFunctions.DataStats import DataStats
from HelperFunctions.DistributePoolingSet import DistributePoolingSet
from HelperFunctions.HelperFunction import HelperFunction
from PrepareData.ParsingLogic import ParsingLogic
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from Utils.LoggerUtil import LoggerUtil


class PrepareDataset:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.db_utils = DBUtils()
        self.parser = ParsingLogic()
        self.dis_pool = DistributePoolingSet()
        self.kmeans = KMeansImpl()
        self.helper = HelperFunction()
        self.config = ConfigUtil().get_config_instance()
        self.data_stats = DataStats()

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
        avclass_collection_name = self.config['environment']['mongo']['avclass_collection_name']

        c2db_collection = db[c2db_collection_name]
        avclass_collection = db[avclass_collection_name]
        return client, c2db_collection, avclass_collection

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
        l = classified_families.values()
        l = self.helper.is_nested_list(l)
        return l

    def generate_feature_pool(self, collection, list_of_keys, config_param_chunk_size, feature_pool_path):
        feature_pool_part_path_list = list()
        count = 0
        iteration = 0
        while count < len(list_of_keys):
            self.log.info("Iteration : {}".format(iteration))
            if count + config_param_chunk_size < len(list_of_keys):
                value = list_of_keys[count:count + config_param_chunk_size]
            else:
                value = list_of_keys[count:]
            count += config_param_chunk_size
            doc2bow = self.parser.parse_each_document(value, collection)
            iteration += 1
            feature_pool_part_path_list_value = self.dis_pool.save_feature_pool(feature_pool_path,
                                                                                doc2bow.values(),
                                                                                iteration)
            feature_pool_part_path_list.append(feature_pool_part_path_list_value)
            del doc2bow
        return feature_pool_part_path_list

    def get_data_as_matrix(self, client, collection,
                           list_of_keys, config_param_chunk_size,
                           feature_pool_path, feature_vector_path):

        feature_pool_part_path_list = self.helper.get_files_ends_with_extension(extension="dump",
                                                                                path=feature_pool_path)
        feature_vector_part_path_list = self.helper.get_files_ends_with_extension(extension="npz",
                                                                                  path=feature_vector_path)

        if len(feature_pool_part_path_list) == math.ceil(len(list_of_keys) * 1.0 / config_param_chunk_size):
            self.log.info("Feature pool already generated at : {}".format(feature_pool_path))
        else:
            feature_pool_part_path_list = self.generate_feature_pool(collection, list_of_keys, config_param_chunk_size,
                                                                     feature_pool_path)

        client.close()

        if len(feature_vector_part_path_list) == math.ceil(len(list_of_keys) * 1.0 / config_param_chunk_size):
            self.log.info("Feature vector already generated at : {}".format(feature_vector_path))
            return feature_vector_path
        else:
            return self.parser.convert2vec(feature_pool_part_path_list, feature_vector_path,
                                           num_rows=len(list_of_keys))

    def load_data(self):
        client, c2db_collection, avclass_collection = self.get_collection()
        cursor = c2db_collection.aggregate([{"$group": {"_id": '$key'}}])
        config_param_chunk_size = self.config["data"]["config_param_chunk_size"]
        list_of_keys = list()

        for each_element in cursor:
            list_of_keys.append(each_element['_id'])

        self.log.info("Total keys before AVClass : {}".format(len(list_of_keys)))
        list_of_keys = self.get_families_data(avclass_collection, list_of_keys, config_param_chunk_size)
        self.log.info("Total keys after AVClass : {}".format(len(list_of_keys)))
        pi.dump(list_of_keys, open(self.config["data"]["list_of_keys"] + "/" + "names.dump", "w"))

        feature_pool_path = self.config['data']['feature_pool_path']
        feature_vector_path = self.config['data']['feature_vector_path']
        self.helper.create_dir_if_absent(feature_pool_path)
        self.helper.create_dir_if_absent(feature_vector_path)
        fv_dist_path_names = self.get_data_as_matrix(client, c2db_collection, list_of_keys, config_param_chunk_size,
                                                     feature_pool_path, feature_vector_path)
        self.data_stats.main()


if __name__ == "__main__":
    prepare_dataset = PrepareDataset()
    prepare_dataset.load_data()
