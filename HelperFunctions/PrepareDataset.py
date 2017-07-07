import urllib
import pickle as pi

from collections import defaultdict
from sklearn.cluster import DBSCAN

from Utils.LoggerUtil import LoggerUtil
from Utils.DBUtils import DBUtils
from DimensionalityReduction.PcaNew import PcaNew
from ParsingLogic import ParsingLogic
from DistributePoolingSet import DistributePoolingSet
from Clustering.KMeansImpl import KMeansImpl
from HelperFunction import HelperFunction


class PrepareDataset:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.db_utils = DBUtils()
        self.parser = ParsingLogic()
        self.dim_red = PcaNew()
        self.dis_pool = DistributePoolingSet()
        self.kmeans = KMeansImpl()
        self.helper = HelperFunction()

    def get_collection(self):
        username = "admin"
        password = urllib.quote("goodDevelopers@123")
        address = "localhost"
        port = "27017"
        auth_db = "admin"

        client = self.db_utils.get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=True,
                                          username=username, password=password)

        db = client['cuckoo']
        collection = db['cluster2db']
        return collection

    def get_families_data(self, collection, list_of_keys):
        entire_families = defaultdict(list)

        for each_key in list_of_keys:
            query = {"feature": "malheur", "key": each_key}
            local_cursor = collection.find(query)
            for each_value in local_cursor:
                key = each_value['key']
                entire_families[each_value['value'][key]["malheur"]["family"]].append(key)

        for key, values in entire_families.items():
            self.log.info("Family : {} \t Variants : {}".format(key, len(values)))

        entire_families.pop("")
        self.log.info("Total Number of families : {} ".format(len(entire_families)))

    def get_data_as_matrix(self, collection, list_of_keys, config_param_chunk_size):
        count = 0
        index = 0
        dist_fnames = self.dis_pool.load_distributed_pool()

        if len(dist_fnames) == 0:
            while count < len(list_of_keys):
                if count + config_param_chunk_size < len(list_of_keys):
                    value = list_of_keys[count:count + config_param_chunk_size]
                else:
                    value = list_of_keys[count:]
                count += config_param_chunk_size
                doc2bow = self.parser.parse_each_document(value, collection)
                dist_fnames.append(self.dis_pool.save_distributed_pool(doc2bow, index))
                index += 1

        # Converting to feature vector
        fv_dist_path_names, num_cols = self.parser.convert2vec(dist_fnames, len(list_of_keys))
        return fv_dist_path_names

    def load_data(self):
        collection = self.get_collection()
        cursor = collection.aggregate([{"$group": {"_id": '$key'}}])
        list_of_keys = list()

        for each_element in cursor:
            list_of_keys.append(each_element['_id'])

        self.get_families_data(collection, list_of_keys)
        # Because the number of samples will always be less than the number of features.
        config_param_chunk_size = len(list_of_keys)
        pi.dump(list_of_keys, open("names.dump", "w"))
        fv_dist_path_names = self.get_data_as_matrix(collection, list_of_keys, config_param_chunk_size)
        reduced_matrix = self.dim_red.prepare_data_for_pca(config_param_chunk_size, fv_dist_path_names)
        self.log.info("Reduced Matrix Shape : {}".format(reduced_matrix.shape))
        dbscan = DBSCAN()
        dbscan.fit(reduced_matrix)
        self.log.info("DBScan labels : {}".format(dbscan.labels_.tolist()))
        kmeans_clusters = self.kmeans.get_clusters_kmeans(reduced_matrix, names=list_of_keys, k=16)
        families = self.kmeans.get_family_names(collection, kmeans_clusters)


if __name__ == "__main__":
    prepare_dataset = PrepareDataset()
    prepare_dataset.load_data()
