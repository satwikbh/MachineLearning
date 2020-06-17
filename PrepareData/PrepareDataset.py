import json
import math
import pickle as pi
from collections import defaultdict
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from time import time
from urllib.parse import quote

import numpy as np

from HelperFunctions.DataStats import DataStats
from HelperFunctions.DistributePoolingSet import DistributePoolingSet
from HelperFunctions.HelperFunction import HelperFunction
from PrepareData.ParsingLogic import ParsingLogic
from PrepareData.TrieBasedPruning import TrieBasedPruning
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from Utils.LoggerUtil import LoggerUtil


class PrepareDataset:
    def __init__(self, use_trie_pruning):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.db_utils = DBUtils()
        self.use_trie_pruning = use_trie_pruning
        self.parser = ParsingLogic(use_trie_pruning=self.use_trie_pruning)
        self.dis_pool = DistributePoolingSet()
        self.helper = HelperFunction()
        self.config = ConfigUtil().get_config_instance()
        self.data_stats = DataStats(use_trie_pruning=self.use_trie_pruning)
        self.trie_based_pruning = TrieBasedPruning()
        self.feature_pool_part_path_list = list()

    def get_collection(self):
        username = self.config['environment']['mongo']['username']
        pwd = self.config['environment']['mongo']['password']
        password = quote(pwd)
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
                self.log.info(F"Iteration : {iteration}")
                if count + config_param_chunk_size < len(list_of_keys):
                    p_value = list_of_keys[count:count + config_param_chunk_size]
                else:
                    p_value = list_of_keys[count:]
                p_value = self.helper.convert_from_vs_keys(p_value)
                count += config_param_chunk_size
                local_cursor = collection.find({"md5": {"$in": p_value}})
                for index, each_value in enumerate(local_cursor):
                    family = each_value['avclass']['result']
                    val = "VirusShare_" + each_value['md5']
                    if "SINGLETON" in family:
                        unclassified_families[family].append(val)
                    else:
                        classified_families[family].append(val)
                iteration += 1
            except Exception as e:
                self.log.error(F"Error : {e}")

        malware_families_path = self.config['data']['malware_families_list']

        json.dump(classified_families, open(malware_families_path + "/" + "classified_families.json", "w"))
        json.dump(unclassified_families, open(malware_families_path + "/" + "unclassified_families.json", "w"))

        self.log.info(F"Classified : {len(classified_families)} \t Unclassified : {len(unclassified_families)}")
        nested_list = classified_families.values()
        nested_list = self.helper.is_nested_list(nested_list)
        return nested_list

    def collect_parallel_result(self, feature_pool_part_path_list_value):
        """
        Collect the result of the parallely executed method
        :param feature_pool_part_path_list_value:
        :return:
        """
        self.feature_pool_part_path_list.append(feature_pool_part_path_list_value)

    def parallel_parse_and_save_docs(self, list_of_keys, collection, feature_pool_path):
        """
        Aggregate over the list of keys for the collection.
        Parse the resulting documents in doc2bow format and write to file.
        While saving the file, we use a random number to counter the negative effects of the parallelization.
        :param list_of_keys:
        :param collection:
        :param feature_pool_path:
        :return:
        """
        doc2bow = self.parser.parse_list_of_documents(list_of_keys, collection)
        index = np.random.RandomState().randint(0, 10 ** 12)
        feature_pool_part_path_list_value = self.dis_pool.save_feature_pool(feature_pool_path,
                                                                            self.helper.dict_values_to_list(
                                                                                doc2bow.values()),
                                                                            index)
        return feature_pool_part_path_list_value

    def parallel_generate_feature_pool(self, collection, list_of_keys, config_param_chunk_size, feature_pool_path):
        """
        This is a parallelized version of the @generate_feature_pool method.
        :param collection:
        :param list_of_keys:
        :param config_param_chunk_size:
        :param feature_pool_path:
        :return:
        """
        pool = Pool(cpu_count())

        meta_p_keys_list = list()
        count = 0
        iteration = 0
        while count < len(list_of_keys):
            self.log.info(F"Iteration : {iteration}")
            if count + config_param_chunk_size < len(list_of_keys):
                p_list_of_keys = list_of_keys[count:count + config_param_chunk_size]
            else:
                p_list_of_keys = list_of_keys[count:]
            meta_p_keys_list.append(p_list_of_keys)
            count += config_param_chunk_size
            iteration += 1

        args = zip(meta_p_keys_list, collection, feature_pool_path)
        pool.starmap_async(self.parallel_parse_and_save_docs, args, callback=self.collect_parallel_result)
        pool.close()
        pool.join()

        # Instead of returning the feature_pool_part_path_list, use self.feature_pool_part_path_list

    def generate_feature_pool(self, collection, list_of_keys, config_param_chunk_size, feature_pool_path):
        feature_pool_part_path_list = list()
        count = 0
        iteration = 0
        while count < len(list_of_keys):
            self.log.info(F"Iteration : {iteration}")
            if count + config_param_chunk_size < len(list_of_keys):
                p_list_of_keys = list_of_keys[count:count + config_param_chunk_size]
            else:
                p_list_of_keys = list_of_keys[count:]
            count += config_param_chunk_size
            doc2bow = self.parser.parse_list_of_documents(p_list_of_keys, collection)
            iteration += 1
            feature_pool_part_path_list_value = self.dis_pool.save_feature_pool(feature_pool_path,
                                                                                self.helper.dict_values_to_list(
                                                                                    doc2bow.values()),
                                                                                iteration)
            feature_pool_part_path_list.append(feature_pool_part_path_list_value)
            del doc2bow
        return feature_pool_part_path_list

    def pruning_method(self, list_of_keys, config_param_chunk_size, feature_pool_path, pruned_indi_feature_pool_path,
                       pruned_feature_vector_path):
        pruned_indi_feature_pool_part_list = self.helper.get_files_ends_with_extension(extension="json",
                                                                                       path=pruned_indi_feature_pool_path)
        feature_vector_part_path_list = self.helper.get_files_ends_with_extension(extension="npz",
                                                                                  path=pruned_feature_vector_path)
        if len(pruned_indi_feature_pool_part_list) == 7:
            self.log.info(F"Feature pool already generated at : {pruned_indi_feature_pool_path}")
        else:
            self.trie_based_pruning.main()
        feature_pool_part_path_list = self.helper.get_files_ends_with_extension(extension="dump",
                                                                                path=feature_pool_path)
        if len(feature_vector_part_path_list) == math.ceil(len(list_of_keys) * 1.0 / config_param_chunk_size):
            self.log.info(F"Feature vector already generated at : {pruned_feature_vector_path}")
            return pruned_feature_vector_path
        else:
            return self.parser.convert2vec(feature_pool_part_path_list=feature_pool_part_path_list,
                                           feature_vector_path=pruned_feature_vector_path, num_rows=len(list_of_keys),
                                           pruned_feature_pool_path=pruned_indi_feature_pool_path)

    def non_pruning_method(self, client, collection, list_of_keys, config_param_chunk_size, feature_pool_path,
                           unpruned_feature_vector_path):
        feature_pool_part_path_list = self.helper.get_files_ends_with_extension(extension="dump",
                                                                                path=feature_pool_path)
        feature_vector_part_path_list = self.helper.get_files_ends_with_extension(extension="npz",
                                                                                  path=unpruned_feature_vector_path)

        if len(feature_pool_part_path_list) == math.ceil(len(list_of_keys) * 1.0 / config_param_chunk_size):
            self.log.info(F"Feature pool already generated at : {feature_pool_path}")
        else:
            feature_pool_part_path_list = self.generate_feature_pool(collection, list_of_keys,
                                                                     config_param_chunk_size,
                                                                     feature_pool_path)
        client.close()

        if len(feature_vector_part_path_list) == math.ceil(len(list_of_keys) * 1.0 / config_param_chunk_size):
            self.log.info(F"Feature vector already generated at : {unpruned_feature_vector_path}")
            return unpruned_feature_vector_path
        else:
            return self.parser.convert2vec(feature_pool_part_path_list=feature_pool_part_path_list,
                                           feature_vector_path=unpruned_feature_vector_path, num_rows=len(list_of_keys))

    def get_data_as_matrix(self, **kwargs):
        client = kwargs["client"]
        collection = kwargs["collection"]
        list_of_keys = kwargs["list_of_keys"]
        config_param_chunk_size = kwargs["config_param_chunk_size"]
        feature_pool_path = kwargs["feature_pool_path"]
        pruned_indi_feature_pool_path = kwargs["pruned_indi_feature_pool_path"]
        pruned_feature_vector_path = kwargs["pruned_feature_vector_path"]
        unpruned_feature_vector_path = kwargs["unpruned_feature_vector_path"]

        if self.use_trie_pruning:
            self.pruning_method(list_of_keys=list_of_keys, config_param_chunk_size=config_param_chunk_size,
                                pruned_indi_feature_pool_path=pruned_indi_feature_pool_path,
                                pruned_feature_vector_path=pruned_feature_vector_path,
                                feature_pool_path=feature_pool_path)
        else:
            self.non_pruning_method(client=client, collection=collection, list_of_keys=list_of_keys,
                                    config_param_chunk_size=config_param_chunk_size,
                                    feature_pool_path=feature_pool_path,
                                    unpruned_feature_vector_path=unpruned_feature_vector_path)

    def generate_labels(self, avclass_collection, list_of_keys, config_param_chunk_size):
        md5_keys = self.helper.convert_from_vs_keys(list_of_keys)
        x = 0
        index_pointer = 0
        key_index_to_family_mapping = defaultdict()
        key_index_to_family_mapping.default_factory = key_index_to_family_mapping.__len__

        list_of_families = defaultdict()
        list_of_families.default_factory = list_of_families.__len__

        checker = list()
        index = 0

        while x < len(md5_keys):
            if x + config_param_chunk_size > len(md5_keys):
                p_keys = md5_keys[x:]
            else:
                p_keys = md5_keys[x:x + config_param_chunk_size]
            query = [
                {"$match": {"md5": {"$in": p_keys}}},
                {"$addFields": {"__order": {"$indexOfArray": [p_keys, "$md5"]}}},
                {"$sort": {"__order": 1}}
            ]
            cursor = avclass_collection.aggregate(query, allowDiskUse=True)
            for _ in cursor:
                try:
                    md5 = _["md5"]
                    family = _["avclass"]["result"]
                    key_index_to_family_mapping[index_pointer] = family
                    index_pointer += 1
                    checker.append(md5)
                except Exception as e:
                    self.log.error(F"Error : {e}")
            x += config_param_chunk_size
            index += 1
            self.log.info(F"Iteration : #{index}")
        if not np.all([md5_keys[x] == checker[x] for x in range(len(md5_keys))]):
            raise Exception("Labels are not generated properly")

        for x in key_index_to_family_mapping.values():
            list_of_families[x]

        labels = [list_of_families[x] for x in key_index_to_family_mapping.values()]
        return labels

    def load_data(self):
        start_time = time()
        config_param_chunk_size = self.config["data"]["config_param_chunk_size"]
        labels_path = self.config["data"]["labels_path"]
        pruned_indi_feature_pool_path = self.config["data"]["pruned_indi_feature_pool_path"]
        unpruned_feature_vector_path = self.config["data"]["unpruned_feature_vector_path"]

        feature_pool_path = self.config["data"]["feature_pool_path"]
        pruned_feature_vector_path = self.config["data"]["pruned_feature_vector_path"]

        client, c2db_collection, avclass_collection = self.get_collection()
        """
        cursor = c2db_collection.aggregate([{"$group": {"_id": '$key'}}], allowDiskUse=True)

        list_of_keys = list()

        for each_element in cursor:
            list_of_keys.append(each_element['_id'])

        """
        list_of_keys = json.load(open("/home/satwik/Documents/MachineLearning/Data346k/list_of_keys.json", "rb"))
        """

        self.log.info(F"Total keys before AVClass : {len(list_of_keys)}")
        list_of_keys = self.get_families_data(avclass_collection, list_of_keys, config_param_chunk_size)
        self.log.info(F"Total keys after AVClass : {len(list_of_keys)}")
        pi.dump(list_of_keys, open(self.config["data"]["list_of_keys"] + "/" + "names.dump", "wb")

        if self.use_trie_pruning:
            self.helper.create_dir_if_absent(pruned_indi_feature_pool_path)
            self.helper.create_dir_if_absent(pruned_feature_vector_path)
        else:
            self.helper.create_dir_if_absent(feature_pool_path)
            self.helper.create_dir_if_absent(unpruned_feature_vector_path)
        self.get_data_as_matrix(client=client, collection=c2db_collection, list_of_keys=list_of_keys,
                                config_param_chunk_size=config_param_chunk_size,
                                feature_pool_path=feature_pool_path,
                                pruned_indi_feature_pool_path=pruned_indi_feature_pool_path,
                                pruned_feature_vector_path=pruned_feature_vector_path,
                                unpruned_feature_vector_path=unpruned_feature_vector_path)
        self.data_stats.main()
        """
        labels = self.generate_labels(avclass_collection, list_of_keys, config_param_chunk_size)
        json.dump(labels, open(labels_path + "/" + "labels.json", "w"))
        self.log.info(F"Total time taken : {time() - start_time}")


if __name__ == "__main__":
    prepare_dataset = PrepareDataset(use_trie_pruning=True)
    prepare_dataset.load_data()
