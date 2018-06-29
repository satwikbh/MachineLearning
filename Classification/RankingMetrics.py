from time import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import kendalltau
from sklearn.externals import joblib

from Adaboost import Adaboost
from DecisionTrees import DecisionTrees
from ExtraTrees import ExtraTrees
from HelperFunctions.HelperFunction import HelperFunction
from NaiveBayesClassifier import NaiveBayesClassifier, CoreClassificationLogic
from RandomForest import RandomForest
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class RankingMetrics:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.helper = HelperFunction()
        self.bnb_clf = NaiveBayesClassifier()
        self.bnb_ccl = CoreClassificationLogic()
        self.rf_clf = RandomForest()
        self.et_clf = ExtraTrees()
        self.dt_clf = DecisionTrees()
        self.adaboost_clf = Adaboost()

    @staticmethod
    def classifier_ranking(clf, test_data, av_test_dist):
        """
        Computes Kendall's Tau and Pearson Correlation Coeff for the test data.
        :param clf:
        :param test_data:
        :param av_test_dist:
        :return:
        """
        kt_list = list()
        pearson_coeff_list = list()
        for index, pred_prob in enumerate(clf.predict_proba(test_data)):
            x, y = av_test_dist[index].toarray(), pred_prob
            kt = kendalltau(x, y)
            kt_list.append(kt)
            pcc = np.corrcoef(x, y)[:, 1]
            pearson_coeff_list.append(pcc)
        return kt_list, pearson_coeff_list

    @staticmethod
    def get_dr_matrices(smote_path):
        """
        Takes the SMOTE data's path, loads and returns them
        :param smote_path:
        :return:
        """
        x_train_smote = load_npz(smote_path + "/" + "smote_train_data.npz")
        y_train_smote = np.load(smote_path + "/" + "smote_train_labels.npz")['arr_0']
        x_test_smote = load_npz(smote_path + "/" + "smote_test_data.npz")
        y_test_smote = np.load(smote_path + "/" + "smote_test_labels.npz")['arr_0']
        av_train_dist = load_npz(smote_path + "/" + "smote_avclass_train_dist.npz")
        av_test_dist = load_npz(smote_path + "/" + "smote_avclass_test_dist.npz")

        return x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_train_dist, av_test_dist

    def bernoulli_nb(self, x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist,
                     ranking_results_path, bnb_model_path):
        """

        :param x_train_smote:
        :param y_train_smote:
        :param x_test_smote:
        :param y_test_smote:
        :param av_test_dist:
        :param ranking_results_path:
        :param bnb_model_path:
        :return:
        """
        self.log.info("******* Bernouille Naive Bayes Classifier *******")
        clf = self.bnb_clf.classification(train_data=x_train_smote, train_labels=y_train_smote)
        y_pred = self.bnb_clf.prediction(clf=clf, test_data=x_test_smote)
        self.bnb_clf.compute_metrics(y_pred=y_pred, y_test=y_test_smote, dr_name="smote",
                                     bnb_results_path=ranking_results_path)
        joblib.dump(clf, bnb_model_path + "/" + "naive_bayes.pkl")
        kt_list, pearson_coeff_list = self.bnb_ccl.bernoulli_nb_classifier_ranking(clf=clf, x_test=x_test_smote,
                                                                                   av_test_dist=av_test_dist)
        kt_df = pd.DataFrame(kt_list)
        self.log.info(kt_df.describe())
        kt_df.to_msgpack(ranking_results_path + "/" + "kt_bnb_smote")

        pcc_df = pd.DataFrame(pearson_coeff_list)
        self.log.info(pcc_df.describe())
        pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_bnb_smote")

    def random_forest(self, x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist,
                      ranking_results_path, rf_model_path):
        """

        :param x_train_smote:
        :param y_train_smote:
        :param x_test_smote:
        :param y_test_smote:
        :param av_test_dist:
        :param ranking_results_path:
        :param rf_model_path:
        :return:
        """
        self.log.info("******* Random Forest Classifier *******")
        clf = self.rf_clf.classification(train_data=x_train_smote, train_labels=y_train_smote)
        y_pred = self.rf_clf.prediction(clf=clf, test_data=x_test_smote)
        self.rf_clf.compute_metrics(y_test=y_test_smote, y_pred=y_pred,
                                    rf_results_path=ranking_results_path, dr_name="smote")
        joblib.dump(clf, rf_model_path + "/" + "random_forest.pkl")
        kt_list, pearson_coeff_list = self.classifier_ranking(clf=clf, test_data=x_test_smote,
                                                              av_test_dist=av_test_dist)
        kt_df = pd.DataFrame(kt_list)
        self.log.info(kt_df.describe())
        kt_df.to_msgpack(ranking_results_path + "/" + "kt_rf_smote")

        pcc_df = pd.DataFrame(pearson_coeff_list)
        self.log.info(pcc_df.describe())
        pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_rf_smote")

    def extra_trees(self, x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, ranking_results_path,
                    et_model_path):
        """

        :param x_train_smote:
        :param y_train_smote:
        :param x_test_smote:
        :param y_test_smote:
        :param av_test_dist:
        :param ranking_results_path:
        :param et_model_path:
        :return:
        """
        self.log.info("******* Extra Trees Classifier *******")
        clf = self.et_clf.classification(train_data=x_train_smote, train_labels=y_train_smote)
        y_pred = self.et_clf.prediction(clf=clf, test_data=x_test_smote)
        self.et_clf.compute_metrics(y_test=y_test_smote, y_pred=y_pred,
                                    et_results_path=ranking_results_path, dr_name="smote")
        joblib.dump(clf, et_model_path + "/" + "extra_trees.pkl")
        kt_list, pearson_coeff_list = self.classifier_ranking(clf=clf, test_data=x_test_smote,
                                                              av_test_dist=av_test_dist)
        kt_df = pd.DataFrame(kt_list)
        self.log.info(kt_df.describe())
        kt_df.to_msgpack(ranking_results_path + "/" + "kt_et_smote")

        pcc_df = pd.DataFrame(pearson_coeff_list)
        self.log.info(pcc_df.describe())
        pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_et_smote")

    def decision_tree(self, x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist,
                      ranking_results_path, dt_model_path):
        """

        :param x_train_smote:
        :param y_train_smote:
        :param x_test_smote:
        :param y_test_smote:
        :param av_test_dist:
        :param ranking_results_path:
        :param dt_model_path:
        :return:
        """
        self.log.info("******* Decision Tree Classifier *******")
        clf = self.dt_clf.classification(train_data=x_train_smote, train_labels=y_train_smote)
        y_pred = self.dt_clf.prediction(clf=clf, test_data=x_test_smote)
        self.dt_clf.compute_metrics(y_test=y_test_smote, y_pred=y_pred,
                                    dt_results_path=ranking_results_path, dr_name="smote")
        joblib.dump(clf, dt_model_path + "/" + "decision_tree.pkl")
        kt_list, pearson_coeff_list = self.classifier_ranking(clf=clf, test_data=x_test_smote,
                                                              av_test_dist=av_test_dist)
        kt_df = pd.DataFrame(kt_list)
        self.log.info(kt_df.describe())
        kt_df.to_msgpack(ranking_results_path + "/" + "kt_dt_smote")

        pcc_df = pd.DataFrame(pearson_coeff_list)
        self.log.info(pcc_df.describe())
        pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_dt_smote")

    def adaboost(self, x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, ranking_results_path,
                 adaboost_model_path):
        """

        :param x_train_smote:
        :param y_train_smote:
        :param x_test_smote:
        :param y_test_smote:
        :param av_test_dist:
        :param ranking_results_path:
        :param adaboost_model_path:
        :return:
        """
        self.log.info("******* Adaboost Classifier *******")
        clf = self.adaboost_clf.classification(train_data=x_train_smote, train_labels=y_train_smote)
        y_pred = self.adaboost_clf.prediction(clf=clf, test_data=x_test_smote)
        self.adaboost_clf.compute_metrics(y_test=y_test_smote, y_pred=y_pred, dr_name="smote",
                                          adaboost_results_path=ranking_results_path)
        joblib.dump(clf, adaboost_model_path + "/" + "adaboost.pkl")
        kt_list, pearson_coeff_list = self.classifier_ranking(clf=clf, test_data=x_test_smote,
                                                              av_test_dist=av_test_dist)
        kt_df = pd.DataFrame(kt_list)
        self.log.info(kt_df.describe())
        kt_df.to_msgpack(ranking_results_path + "/" + "kt_adaboost_smote")

        pcc_df = pd.DataFrame(pearson_coeff_list)
        self.log.info(pcc_df.describe())
        pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_adaboost_smote")

    def train_classifiers(self, **kwargs):
        x_train_smote = kwargs["x_train_smote"]
        y_train_smote = kwargs["y_train_smote"]
        x_test_smote = kwargs["x_test_smote"]
        y_test_smote = kwargs["y_test_smote"]
        av_test_dist = kwargs["av_test_dist"]
        bnb_model_path = kwargs["bnb_model_path"]
        rf_model_path = kwargs["rf_model_path"]
        et_model_path = kwargs["et_model_path"]
        dt_model_path = kwargs["dt_model_path"]
        adaboost_model_path = kwargs["adaboost_model_path"]
        ranking_results_path = kwargs["ranking_results_path"]

        self.bernoulli_nb(x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, ranking_results_path,
                          bnb_model_path)
        self.random_forest(x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, ranking_results_path,
                           rf_model_path)
        self.extra_trees(x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, ranking_results_path,
                         et_model_path)
        self.decision_tree(x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, ranking_results_path,
                           dt_model_path)
        self.adaboost(x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, ranking_results_path,
                      adaboost_model_path)

    def main(self):
        start_time = time()
        smote_path = self.config["data"]["smote_data"]
        ranking_results_path = self.config["data"]["ranking_results_path"]
        bnb_model_path = self.config["models"]["naive_bayes"]["model_path"]
        rf_model_path = self.config["models"]["random_forest"]["model_path"]
        et_model_path = self.config["models"]["extra_trees"]["model_path"]
        dt_model_path = self.config["models"]["decision_trees"]["model_path"]
        adaboost_model_path = self.config["models"]["adaboost"]["model_path"]

        x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_train_dist, av_test_dist = self.get_dr_matrices(
            smote_path=smote_path)
        self.train_classifiers(x_train_smote=x_train_smote, y_train_smote=y_train_smote, x_test_smote=x_test_smote,
                               y_test_smote=y_test_smote, av_test_dist=av_test_dist, bnb_model_path=bnb_model_path,
                               rf_model_path=rf_model_path, et_model_path=et_model_path, dt_model_path=dt_model_path,
                               adaboost_model_path=adaboost_model_path, ranking_results_path=ranking_results_path)
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    rm = RankingMetrics()
    rm.main()
