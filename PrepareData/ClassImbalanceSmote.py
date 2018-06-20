import json
import numpy as np

from time import time

from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, vstack, save_npz

from imblearn.over_sampling import SMOTE

from Utils.LoggerUtil import LoggerUtil
from Utils.DBUtils import DBUtils
from Utils.ConfigUtil import ConfigUtil
from HelperFunctions.HelperFunction import HelperFunction


class ClassImbalanceSmote:

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.db_utils = DBUtils()
        self.helper = HelperFunction()

    def get_vector(self, keys, docs):
        vector = [0] * len(keys)
        for each_doc in docs:
            family_name = each_doc["family_name"]
            if family_name in keys:
                vector[keys.index(family_name)] = 1

        if vector.count(1) == 1:
            pass
        elif vector.count(1) < 1:
            self.log.error("Error : {}".format(docs))
        else:
            for each_doc in docs:
                family_name = each_doc["family_name"]
                score = each_doc["score"]
                if family_name in keys:
                    vector[keys.index(family_name)] = 1 * score
        data = list()
        column = list()
        for index, value in enumerate(vector):
            if value != 0:
                column.append(index)
                data.append(value)
        row = len(column) * [0]
        matrix = coo_matrix((data, (row, column)), shape=(1, 86), dtype=float)
        return matrix

    def prepare_avclass_dist(self, avclass_dist_meta, classified):
        keys = classified.keys()
        avclass_dist = list()

        for key, list_of_vs_keys in classified.items():
            self.log.info("Working on Malware family : {}\tNumber of keys : {}".format(key, len(list_of_vs_keys)))
            family_vector = list()
            for vs_name in list_of_vs_keys:
                md5 = vs_name.split("_")[1]
                value = avclass_dist_meta[md5]
                vec = self.get_vector(keys=keys, docs=value)
                family_vector.append(vec)
            family_vector = vstack(family_vector)
            avclass_dist.append(family_vector)

        avclass_dist = vstack(avclass_dist)
        self.log.info("Final Shape : {}".format(avclass_dist.shape))
        return avclass_dist

    @staticmethod
    def perform_smote(data, labels, avclass_dist, n_jobs):
        x_train, x_test, y_train, y_test, av_train_dist, av_test_dist = train_test_split(data, labels,
                                                                                         avclass_dist, test_size=0.33,
                                                                                         random_state=11,
                                                                                         stratify=labels)
        smote = SMOTE()
        smote.n_jobs = n_jobs
        x_train_smote, y_train_smote = smote.fit_sample(x_test, y_test)
        return x_train_smote, y_train_smote, x_train, y_train, av_train_dist, av_test_dist

    def save_avclass_distribution(self, avclass_dist_path, avclass_dist):
        try:
            file_name = avclass_dist_path + "/" + "avclass_dist.npz"
            file_object = open(file_name, "w")
            save_npz(file_object, avclass_dist, compressed=True)
            file_object.close()
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def save_smote_data(self, smote_path, x_train_smote, y_train_smote,
                        x_test_smote, y_test_smote,
                        av_train_dist, av_test_dist):
        try:
            np.savez_compressed(smote_path + "/" + "smote_train_data", x_train_smote)
            np.savez_compressed(smote_path + "/" + "smote_train_labels", y_train_smote)

            np.savez_compressed(smote_path + "/" + "smote_test_data", x_test_smote)
            np.savez_compressed(smote_path + "/" + "smote_test_labels", y_test_smote)

            np.savez_compressed(smote_path + "/" + "smote_avclass_train_dist", av_train_dist)
            np.savez_compressed(smote_path + "/" + "smote_avclass_test_dist", av_test_dist)
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def main(self):
        start_time = time()
        n_jobs = self.config["environment"]["n_cores"]
        smote_path = self.config["data"]["smote"]
        avclass_dist_path = self.config["data"][""]
        avclass_dist_meta = json.load(open("/tmp/DataMain/avclass_distribution.json"))
        classified = json.load(open("/tmp/DataMain/classified_families.json"))
        avclass_dist = self.prepare_avclass_dist(avclass_dist_meta=avclass_dist_meta, classified=classified)
        self.save_avclass_distribution(avclass_dist=avclass_dist, avclass_dist_path=avclass_dist_path)
        x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_train_dist, av_test_dist = self.perform_smote(
            data=[],
            labels=[],
            avclass_dist=avclass_dist,
            n_jobs=n_jobs)
        self.save_smote_data(smote_path, x_train_smote, y_train_smote,
                             x_test_smote, y_test_smote,
                             av_train_dist, av_test_dist)
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    class_imb_smote = ClassImbalanceSmote()
    class_imb_smote.main()
