import urllib
import pickle as pi
import math
import json

from collections import defaultdict

from LinearDimensionalityReduction.IterativePca import IterativePca
from HelperFunctions.ParsingLogic import ParsingLogic
from HelperFunctions.DistributePoolingSet import DistributePoolingSet
from Clustering.KMeansImpl import KMeansImpl
from HelperFunctions.HelperFunction import HelperFunction
from HelperFunctions.DataStats import DataStats

from Utils.LoggerUtil import LoggerUtil
from Utils.DBUtils import DBUtils
from Utils.ConfigUtil import ConfigUtil


class PrepareDataset:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.db_utils = DBUtils()
        self.parser = ParsingLogic()
        self.dim_red = IterativePca()
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
        entire_families = defaultdict(list)
        classified_families = defaultdict(list)
        unclassified_families = defaultdict(list)

        # FIXME : IMPORTANT
        # This will ensure that we work only on the data which is given a label by AVClass.
        # To revert it return the list_of_keys instead of new_list_of_keys.
        new_list_of_keys = list()

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

        self.log.info("Total Number of families : {} \n"
                      "Classified : {}\n"
                      "Unclassified : {}\n".format(len(entire_families),
                                                   len(classified_families),
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

        feature_pool_part_path_list = self.helper.get_files_ends_with_extension(extension="dump", path=feature_pool_path)
        feature_vector_part_path_list = self.helper.get_files_ends_with_extension(extension="hkl",
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

        # rows, columns = hkl.load(open(fv_dist_path_names[0])).shape
        # rows = len(fv_dist_path_names) * rows
        # self.log.info("Final Matrix shape : {}".format(rows, columns))

        # reduced_matrix = self.dim_red.prepare_data_for_pca(config_param_chunk_size, fv_dist_path_names)
        # self.log.info("Reduced Matrix Shape : {}".format(reduced_matrix.shape))
        # dbscan = DBSCAN()
        # dbscan.fit(reduced_matrix)
        # self.log.info("DBScan labels : {}".format(dbscan.labels_.tolist()))
        # kmeans_clusters = self.kmeans.get_clusters_kmeans(reduced_matrix, names=list_of_keys, k=16)
        # families = self.kmeans.get_family_names(collection, kmeans_clusters)


if __name__ == "__main__":
    prepare_dataset = PrepareDataset()
    prepare_dataset.load_data()
