import pickle as pi
import hickle as hkl
import math
import os
import urllib

from collections import defaultdict
from scipy.sparse import vstack
from time import time
from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils


class AvclassDataPreparation:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.db_utils = DBUtils()

    def get_connection(self):
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
        avclass_collection_name = self.config['environment']['mongo']['avclass_collection_name']

        db = client[db_name]
        avclass_collection = db[avclass_collection_name]
        return client, avclass_collection

    @staticmethod
    def nearest_power_of_two(shape):
        value = 1 << (shape - 1).bit_length()
        return int(math.log(value, 2))

    @staticmethod
    def get_files_ends_with_extension(extension, path):
        all_files = list()
        for each_file in os.listdir(path):
            if each_file.endswith(extension):
                all_files.append(os.path.join(path, each_file))
        return all_files

    @staticmethod
    def open_files(list_of_files):
        matrix = list()
        for each in list_of_files:
            fv = hkl.load(each)
            matrix.append(fv)
        return matrix

    @staticmethod
    def cursor2list(cursor):
        list_of_keys = list()
        for each in cursor:
            list_of_keys.append(each["_id"])
        return list_of_keys

    def prepare_data(self, classified_indices, unclassified_indices):
        pruned_fv_path = self.config["data"]["pruned_feature_vector_path"]
        list_of_files = self.get_files_ends_with_extension(".hkl", pruned_fv_path)
        matrix = self.open_files(list_of_files)

        # Only the first 25000 variants whose family is determined i.e not SINGLETON is picked.
        counter = 25000
        count = 0
        final_classified_matrix = list()
        final_unclassified_matrix = list()

        for index, key in classified_indices:
            if count >= counter:
                break
            subscript, row_ptr = index / 1000, index % 1000
            curr_matrix = matrix[subscript]
            final_classified_matrix.append(curr_matrix[row_ptr])
            count += 1

        for index, key in unclassified_indices:
            if count <= 0:
                break
            subscript, row_ptr = index / 1000, index % 1000
            curr_matrix = matrix[subscript]
            final_unclassified_matrix.append(curr_matrix[row_ptr])
            count -= 1

        classified_matrix = vstack(final_classified_matrix)
        unclassified_matrix = vstack(final_unclassified_matrix)

        nearest_repr = self.nearest_power_of_two(classified_matrix.shape[1])
        print("Input matrix dimension : {}\tNearest power of 2 : {}".format(classified_matrix.shape, nearest_repr))

        file1 = open("/home/satwik/Documents/MachineLearning/Data/temp" + "/" + "labelled_input_matrix.hkl", "w")
        file2 = open("/home/satwik/Documents/MachineLearning/Data/temp" + "/" + "labelled_input_matrix.hkl", "w")
        hkl.dump(classified_matrix, file1)
        hkl.dump(unclassified_matrix, file2)
        file1.close()
        file2.close()

        return classified_matrix, unclassified_matrix

    def prepare_labels(self, list_of_keys, collection):
        variant_labels = defaultdict(list)
        cursor = collection.aggregate([{"$group": {"_id": "$md5"}}])

        list_of_md5 = self.cursor2list(cursor)

        for index, each_key in enumerate(list_of_md5):
            if index % 1000 == 0:
                self.log.info("Iteration : #{}".format(index / 1000))
            query = {"md5": each_key}
            doc = collection.find(query).next()
            family = doc["avclass"]["result"]
            variant_labels[each_key].append(family)

        classified_indices, unclassified_indices = list(), list()

        for index, each_key in enumerate(list_of_keys):
            tup = (index, each_key)
            if "SINGLETON" in variant_labels[each_key]:
                unclassified_indices.append(tup)
            else:
                classified_indices.append(tup)
        self.log.info(
            "Total number of classified : {}\nTotal number of unclassified : {}".format(len(classified_indices),
                                                                                        len(unclassified_indices)))
        return classified_indices, unclassified_indices

    def prepare_avclass_validation_data(self, avclass_collection):
        names_path = self.config["data"]["list_of_keys"]
        list_of_keys = pi.load(open(names_path + "/" + "names.dump"))

        classified_indices, unclassified_indices = self.prepare_labels(list_of_keys, avclass_collection)
        classified_matrix, unclassified_matrix = self.prepare_data(classified_indices, unclassified_indices)
        return classified_matrix, unclassified_matrix

    @staticmethod
    def labels2clusters(labels):
        clusters = defaultdict(list)
        for index, value in enumerate(labels):
            clusters[value].append(index)
        return clusters

    def main(self):
        """
        The labels are sent as input. The output is each cluster with its accuracy and labels.
        :param labels:
        :return:
        """
        start_time = time()
        client, avclass_collection = self.get_connection()
        classified_matrix, unclassified_matrix = self.prepare_avclass_validation_data(avclass_collection)
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    validation = AvclassDataPreparation()
    validation.main()
