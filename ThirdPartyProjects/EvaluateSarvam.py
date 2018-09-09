import numpy as np
import pickle as pi
import urllib

from time import time
from collections import defaultdict
from sklearn.externals import joblib

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from HelperFunctions.HelperFunction import HelperFunction


class EvaluateSarvam:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.db_utils = DBUtils()
        self.helper = HelperFunction()
        self.meta_dict = defaultdict()
        self.sarvam = defaultdict()

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

        sarvam_coll_name = self.config['environment']['mongo']['sarvam_coll_name']
        sarvam_collection = db[sarvam_coll_name]

        avclass_collection_name = self.config['environment']['mongo']['avclass_collection_name']
        avclass_collection = db[avclass_collection_name]

        return sarvam_collection, avclass_collection

    def get_freq_dict(self, document):
        verbose = document["avclass"]["verbose"]
        md5 = document["md5"]
        if len(verbose) == 1:
            family_name, score = verbose[0]
            self.meta_dict[md5] = [{"family_name": family_name, "score": 1}]
        else:
            tmp = list()
            total_score = 0
            for inner_list in verbose:
                family_name, score = inner_list
                total_score += score
            for inner_list in verbose:
                family_name, score = inner_list
                tmp.append({"family_name": family_name, "score": (score * 1.0) / total_score})
            self.meta_dict[md5] = tmp

    def get_avclass_dist(self, list_of_keys, avclass_collection):
        count = 0
        index = 0
        chunk_size = 1000

        while count < len(list_of_keys):
            self.log.info("Working on Iter : #{}".format(index))
            if count + chunk_size < len(list_of_keys):
                p_keys = list_of_keys[count: count + chunk_size]
            else:
                p_keys = list_of_keys[count:]

            query = [
                {"$match": {"md5": {"$in": p_keys}}},
                {"$project": {"avclass.verbose": 1, "md5": 1}},
                {"$addFields": {"__order": {"$indexOfArray": [p_keys, "$md5"]}}},
                {"$sort": {"__order": 1}}
            ]
            cursor = avclass_collection.aggregate(query)
            for doc in cursor:
                self.get_freq_dict(doc)
            count += chunk_size
            index += 1

    @staticmethod
    def get_list_of_binaries(sarvam_collection):
        list_of_binaries = list()
        cursor = sarvam_collection.aggregate([{"$group": {"_id": '$binary'}}])
        for _ in cursor:
            list_of_binaries.append(_["_id"])
        return list_of_binaries

    def sarvam_binary_predictions(self, list_of_binaries, sarvam_collection):
        binary_predictions = defaultdict()
        count = 0
        index = 0
        chunk_size = 1000

        while count < len(list_of_binaries):
            self.log.info("Working on Iter : #{}".format(index))
            if count + chunk_size < len(list_of_binaries):
                p_keys = list_of_binaries[count: count + chunk_size]
            else:
                p_keys = list_of_binaries[count:]

            query = [
                {"$match": {"binary": {"$in": p_keys}}},
                {"$project": {"feature": 1, "binary": 1}},
                {"$addFields": {"__order": {"$indexOfArray": [p_keys, "$md5"]}}},
                {"$sort": {"__order": 1}}
            ]
            cursor = sarvam_collection.aggregate(query)
            for doc in cursor:
                binary = doc["binary"]
                feature = doc["feature"]
                binary_predictions[binary] = feature
            count += chunk_size
            index += 1
        return binary_predictions

    @staticmethod
    def compute_acc(binary_family, binary_values):
        tp = defaultdict()
        for x in binary_family:
            tp[x["family_name"]] = 0
            for y in binary_values:
                if x["family_name"] in [_["family_name"] for _ in y]:
                    tp[x["family_name"]] = 1
        return max(tp.values())

    def evaluate_sarvam(self, ball_tree_model_path, binary_predictions, top_k):
        meta_acc = list()
        failed = list()
        ball_tree_model = joblib.load(ball_tree_model_path + "/" + "bt_model.pkl")
        binary_index = defaultdict()
        binary_index.default_factory = binary_index.__len__()
        for index, value in enumerate(binary_predictions.keys()):
            binary_index[index] = value

        for binary, feature in binary_predictions.items():
            if binary in self.meta_dict:
                dist, ind = ball_tree_model.query([feature], k=top_k)
                binary_family = self.meta_dict[binary.split("VirusShare_")[1]]
                binary_values = list()
                for _ in ind[0]:
                    binary_values.append(self.meta_dict[binary_index[_].split("VirusShare_")[1]])
                num = self.compute_acc(binary_family, binary_values)
                meta_acc.append(num)
            else:
                failed.append(binary)
        self.log.info("Accuracy at top k : {} is : {}".format(top_k, np.mean(meta_acc)))
        return meta_acc, failed

    def main(self):
        start_time = time()
        ball_tree_model_path = self.config["sarvam"]["bt_model_path"]
        sarvam_collection, avclass_collection = self.get_collection()
        list_of_binaries = self.get_list_of_binaries(sarvam_collection)
        list_of_keys = self.helper.convert_from_vs_keys(list_of_vs_keys=list_of_binaries)
        self.get_avclass_dist(list_of_keys=list_of_keys, avclass_collection=avclass_collection)
        binary_predictions = self.sarvam_binary_predictions(list_of_binaries, sarvam_collection)
        meta_acc, failed = self.evaluate_sarvam(ball_tree_model_path, binary_predictions, top_k=5)
        pi.dump(failed, open("failed.pkl", "w"))
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    evaluate = EvaluateSarvam()
    evaluate.main()
