import numpy as np
import pickle as pi

from time import time
from urllib.parse import quote
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from HelperFunctions.HelperFunction import HelperFunction


class EvaluateSarvam:
    def __init__(self, malevol=True):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.db_utils = DBUtils()
        self.helper = HelperFunction()
        self.meta_dict = defaultdict()
        self.sarvam = defaultdict()
        self.malevol = malevol

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

    def compute_metrics(self, binary_family, binary_values, families_list):
        true_positives = list()
        all_positives = list()
        for x in binary_family:
            true_positives.append([x["family_name"]])
            for y in binary_values:
                all_positives.append([_["family_name"] for _ in y])
        acc = self.compute_acc(true_positives=binary_family, all_positives=binary_values)
        if acc == 0:
            precision = 0
            recall = 0
        else:
            precision, recall = self.compute_precision_recall(true_positives=true_positives,
                                                              all_positives=all_positives,
                                                              families_list=families_list)
        return acc, precision, recall

    @staticmethod
    def compute_acc(true_positives, all_positives):
        for x in true_positives:
            if x in [_ for _ in all_positives]:
                return 1
        return 0

    @staticmethod
    def compute_precision_recall(true_positives, all_positives, families_list):
        conf_mat = np.zeros(shape=(len(families_list), len(families_list)))

        for x in true_positives:
            for y in all_positives:
                for z in y:
                    i, j = families_list[x], families_list[z]
                    conf_mat[i][j] += 1

        precision_list, recall_list = list(), list()

        for i, j in conf_mat.shape:
            precision_at_i = conf_mat[i][j] * 1.0 / conf_mat[:, j]
            recall_at_i = conf_mat[i][j] * 1.0 / conf_mat[i, :]
            precision_list.append(precision_at_i)
            recall_list.append(recall_at_i)

        return np.mean(precision_list), np.mean(recall_list)

    def get_malevol_families_list(self, x_test, y_test, ball_tree_model, top_k, binary_index):
        families_list = defaultdict()
        families_list.default_factory = families_list.__len__()

        for index, feature in enumerate(x_test):
            binary = y_test[index].split("VirusShare_")[1]
            if binary in self.meta_dict:
                dist, ind = ball_tree_model.query([feature], k=top_k)
                binary_family = self.meta_dict[binary]
                binary_values = list()
                for _ in ind[0]:
                    t = binary_index[_].split("VirusShare_")[1]
                    if t in self.meta_dict:
                        binary_values.append(self.meta_dict[t])
                [families_list[x["family_name"]] for x in binary_family]
                [families_list[y["family_name"]] for x in binary_values for y in x]

        return families_list

    def evaluate_malevol(self, ball_tree_model, binary_predictions, x_test, y_test, top_k):
        meta_acc = list()
        meta_precision = list()
        meta_recall = list()
        failed = list()

        binary_index = defaultdict()
        binary_index.default_factory = binary_index.__len__()

        for index, value in enumerate(binary_predictions.keys()):
            binary_index[index] = value

        families_list = self.get_malevol_families_list(x_test=x_test,
                                                       y_test=y_test,
                                                       ball_tree_model=ball_tree_model,
                                                       top_k=top_k,
                                                       binary_index=binary_index)

        for index, feature in enumerate(x_test):
            binary = y_test[index].split("VirusShare_")[1]
            if binary in self.meta_dict:
                dist, ind = ball_tree_model.query([feature], k=top_k)
                binary_family = self.meta_dict[binary]
                binary_values = list()
                for _ in ind[0]:
                    t = binary_index[_].split("VirusShare_")[1]
                    if t in self.meta_dict:
                        binary_values.append(self.meta_dict[t])
                acc, precision, recall = self.compute_metrics(binary_family=binary_family,
                                                              binary_values=binary_values,
                                                              families_list=families_list)
                meta_acc.append(acc)
                meta_precision.append(precision)
                meta_recall.append(recall)
            else:
                failed.append(binary)
        self.log.info("Accuracy at top k : {} is : {}".format(top_k, np.mean(meta_acc)))
        self.log.info("Precision and Recall at top k : {} is : {}, {}".format(top_k,
                                                                              np.mean(meta_precision),
                                                                              np.mean(meta_recall)))
        return meta_acc, failed

    def get_sarvam_families_list(self, binary_predictions, ball_tree_model, top_k, binary_index):
        families_list = defaultdict()
        families_list.default_factory = families_list.__len__()

        for binary, feature in binary_predictions.items():
            binary = binary.split("VirusShare_")[1]
            if binary in self.meta_dict:
                dist, ind = ball_tree_model.query([feature], k=top_k)
                binary_family = self.meta_dict[binary]
                binary_values = list()
                for _ in ind[0]:
                    t = binary_index[_].split("VirusShare_")[1]
                    if t in self.meta_dict:
                        binary_values.append(self.meta_dict[t])
                [families_list[x["family_name"]] for x in binary_family]
                [families_list[y["family_name"]] for x in binary_values for y in x]
        return families_list

    def evaluate_sarvam(self, ball_tree_model_path, binary_predictions, top_k):
        meta_acc = list()
        meta_precision = list()
        meta_recall = list()
        failed = list()

        ball_tree_model = joblib.load(ball_tree_model_path + "/" + "bt_model.pkl")
        binary_index = defaultdict()
        binary_index.default_factory = binary_index.__len__()

        for index, value in enumerate(binary_predictions.keys()):
            binary_index[index] = value

        families_list = self.get_sarvam_families_list(binary_predictions=binary_predictions,
                                                      ball_tree_model=ball_tree_model,
                                                      top_k=top_k,
                                                      binary_index=binary_index)

        for binary, feature in binary_predictions.items():
            binary = binary.split("VirusShare_")[1]
            if binary in self.meta_dict:
                dist, ind = ball_tree_model.query([feature], k=top_k)
                binary_family = self.meta_dict[binary]
                binary_values = list()
                for _ in ind[0]:
                    t = binary_index[_].split("VirusShare_")[1]
                    if t in self.meta_dict:
                        binary_values.append(self.meta_dict[t])
                acc, precision, recall = self.compute_metrics(binary_family=binary_family,
                                                              binary_values=binary_values,
                                                              families_list=families_list)
                meta_acc.append(acc)
                meta_precision.append(precision)
                meta_recall.append(recall)
            else:
                failed.append(binary)
        self.log.info("Accuracy at top k : {} is : {}".format(top_k, np.mean(meta_acc)))
        self.log.info("Precision and Recall at top k : {} is : {}, {}".format(top_k,
                                                                              np.mean(meta_precision),
                                                                              np.mean(meta_recall)))
        return meta_acc, failed

    def update_model(self, final_corpus, ball_tree_model_path):
        """
        Update the BallTree model to include all the instances encountered.
        :param final_corpus:
        :param ball_tree_model_path:
        :return:
        """
        self.log.info("Creating Ball Tree for Corpus")
        corpus = np.asarray([np.asarray(document) for document in final_corpus])
        ball_tree = BallTree(corpus)
        self.log.info("Saving Ball Tree model at the following path : {}".format(ball_tree_model_path))
        joblib.dump(ball_tree, ball_tree_model_path + "/" + "bt_model.pkl")
        return ball_tree

    def validation(self, final_corpus, ball_tree_model_path):
        """
        Split the dataset into train and test.
        Then create a model for train and run the query the test.
        :param final_corpus:
        :param ball_tree_model_path:
        :return:
        """
        self.log.info("Splitting final corpus into train and test")
        x_train, x_test, y_train, y_test = train_test_split(final_corpus.values(), final_corpus.keys(), test_size=0.33)
        self.log.info("Creating Ball Tree for Corpus")
        corpus = np.asarray([np.asarray(document) for document in x_train])
        ball_tree = BallTree(corpus)
        self.log.info("Saving Ball Tree model at the following path : {}".format(ball_tree_model_path))
        joblib.dump(ball_tree, ball_tree_model_path + "/" + "bt_model.pkl")
        return ball_tree, x_train, x_test, y_train, y_test

    def main(self):
        start_time = time()
        ball_tree_model_path = self.config["sarvam"]["bt_model_path"]
        malevol_keys_path = self.config["data"]["list_of_keys"]
        sarvam_collection, avclass_collection = self.get_collection()
        if self.malevol:
            list_of_binaries = pi.load(open(malevol_keys_path + "/" + "list_of_keys.pkl"))
        else:
            list_of_binaries = self.get_list_of_binaries(sarvam_collection)
        list_of_keys = self.helper.convert_from_vs_keys(list_of_vs_keys=list_of_binaries)
        self.get_avclass_dist(list_of_keys=list_of_keys, avclass_collection=avclass_collection)
        binary_predictions = self.sarvam_binary_predictions(list_of_binaries, sarvam_collection)
        if self.malevol:
            ball_tree, x_train, x_test, y_train, y_test = self.validation(binary_predictions, ball_tree_model_path)
            meta_acc, failed = self.evaluate_malevol(ball_tree_model=ball_tree, binary_predictions=binary_predictions,
                                                     x_test=x_test, y_test=y_test, top_k=5)
        else:
            meta_acc, failed = self.evaluate_sarvam(ball_tree_model_path, binary_predictions, top_k=5)
        pi.dump(failed, open("failed.pkl", "w"))
        self.update_model(ball_tree_model_path=ball_tree_model_path, final_corpus=binary_predictions.values())
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    evaluate = EvaluateSarvam(malevol=True)
    evaluate.main()
