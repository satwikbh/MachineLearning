import urllib
import pickle as pi
import hickle as hkl
import numpy as np
import json

from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

from DimensionalityReduction.PcaNew import PcaNew
from ParsingLogic import ParsingLogic
from DistributePoolingSet import DistributePoolingSet
from Clustering.KMeansImpl import KMeansImpl
from HelperFunction import HelperFunction
from DataStats import DataStats

from Utils.LoggerUtil import LoggerUtil
from Utils.DBUtils import DBUtils
from Utils.ConfigUtil import ConfigUtil


class PrepareDataset:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.db_utils = DBUtils()
        self.parser = ParsingLogic()
        self.dim_red = PcaNew()
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
        collection_name = self.config['environment']['mongo']['collection_name']
        collection = db[collection_name]
        return client, collection

    def get_families_data(self, collection, list_of_keys):
        entire_families = defaultdict(list)

        for each_key in list_of_keys:
            query = {"feature": "malheur", "key": each_key}
            local_cursor = collection.find(query)
            for each_value in local_cursor:
                key = each_value['key']
                entire_families[each_value['value'][key]["malheur"]["family"]].append(key)

        malware_families_path = self.config['data']['malware_families_list'] + "/" + "malware_families.json"
        entire_families["UNCLASSIFIED"] = entire_families.get("")
        entire_families.pop("")
        json.dump(entire_families, open(malware_families_path, "w"))
        self.log.info("Total Number of families : {} ".format(len(entire_families)))

    def get_data_as_matrix(self, collection, list_of_keys, config_param_chunk_size, feature_pool_path):
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
            values = np.asarray(doc2bow.values())
            iteration += 1
            file_name = open(feature_pool_path + "/" + "feature_pool_part_" + str(iteration) + ".hkl")
            hkl.dump(values, file_name)
            file_name.close()
            del doc2bow

        vec = CountVectorizer(analyzer="word", tokenizer=lambda text: text, binary=True)
        feature_vector = vec.fit_transform(values)
        self.log.info("Sparse Matrix Shape : {}".format(feature_vector.shape))
        file_name = self.dis_pool.save_feature_vector(feature_vector=feature_vector)
        return file_name

    def load_data(self):
        client, collection = self.get_collection()
        cursor = collection.aggregate([{"$group": {"_id": '$key'}}])
        list_of_keys = list()

        for each_element in cursor:
            list_of_keys.append(each_element['_id'])

        self.get_families_data(collection, list_of_keys)
        # Because the number of samples will always be less than the number of features.
        config_param_chunk_size = self.config["data"]["config_param_chunk_size"]
        pi.dump(list_of_keys, open(self.config["data"]["list_of_keys"] + "/" + "names.dump", "w"))

        feature_pool_path = self.config['data']['feature_pool_path']
        fv_dist_path_names = self.get_data_as_matrix(collection, list_of_keys, config_param_chunk_size,
                                                     feature_pool_path)

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
        client.close()
        self.data_stats.main()


if __name__ == "__main__":
    prepare_dataset = PrepareDataset()
    prepare_dataset.load_data()
