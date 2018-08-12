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
from StochasticGradientDescent import StochasticGradientDescent
from Bagging import Bagging
from XGBoostClassifier import XGBoostClassifier

from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class RankingMetrics:
    def __init__(self, use_pruned_data, compute_search_ranking):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.helper = HelperFunction()
        self.compute_search_ranking = compute_search_ranking
        self.bnb_clf = NaiveBayesClassifier(use_pruned_data)
        self.bnb_ccl = CoreClassificationLogic()
        self.rf_clf = RandomForest(use_pruned_data)
        self.et_clf = ExtraTrees(use_pruned_data)
        self.dt_clf = DecisionTrees(use_pruned_data)
        self.adaboost_clf = Adaboost(use_pruned_data)
        self.sgd_clf = StochasticGradientDescent(use_pruned_data)
        self.bagging_clf = Bagging(use_pruned_data)
        self.xgboost_clf = XGBoostClassifier(use_pruned_data)

    @staticmethod
    def classifier_ranking(clf, test_data, av_train_dist):
        """
        Computes Kendall's Tau and Pearson Correlation Coeff for the test data.
        :param clf:
        :param test_data:
        :param av_train_dist:
        :return:
        """
        kt_list = list()
        pearson_coeff_list = list()
        for index, pred_prob in enumerate(clf.predict_proba(test_data)):
            x, y = av_train_dist[index].toarray(), pred_prob
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

    def bernoulli_nb(self, **kwargs):
        x_train = kwargs['x_train']
        y_train = kwargs['y_train']
        x_test = kwargs['x_test']
        y_test = kwargs['y_test']
        av_train_dist = kwargs['av_train_dist']
        ranking_results_path = kwargs['ranking_results_path']
        dr_name = kwargs['dr_name']
        bnb_model_path = kwargs['bnb_model_path']

        self.log.info("******* Bernouille Naive Bayes Classifier *******")
        clf = self.bnb_clf.classification(train_data=x_train, train_labels=y_train)
        y_pred = self.bnb_clf.prediction(clf=clf, test_data=x_test)
        self.bnb_clf.compute_metrics(y_pred=y_pred, y_test=y_test, dr_name=dr_name,
                                     bnb_results_path=ranking_results_path)
        joblib.dump(clf, bnb_model_path + "/" + "naive_bayes" + "_" + dr_name + ".pkl")
        if self.compute_search_ranking:
            kt_list, pearson_coeff_list = self.bnb_ccl.bernoulli_nb_classifier_ranking(clf=clf, x_test=x_test,
                                                                                       av_train_dist=av_train_dist)
            kt_df = pd.DataFrame(kt_list)
            self.log.info(kt_df.describe())
            kt_df.to_msgpack(ranking_results_path + "/" + "kt_bnb" + "_" + dr_name)

            pcc_df = pd.DataFrame(pearson_coeff_list)
            self.log.info(pcc_df.describe())
            pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_bnb" + "_" + dr_name)

    def random_forest(self, **kwargs):
        x_train = kwargs['x_train']
        y_train = kwargs['y_train']
        x_test = kwargs['x_test']
        y_test = kwargs['y_test']
        av_train_dist = kwargs['av_train_dist']
        dr_name = kwargs['dr_name']
        ranking_results_path = kwargs['ranking_results_path']
        rf_model_path = kwargs['rf_model_path']

        self.log.info("******* Random Forest Classifier *******")
        clf = self.rf_clf.classification(train_data=x_train, train_labels=y_train)
        y_pred = self.rf_clf.prediction(clf=clf, test_data=x_test)
        self.rf_clf.compute_metrics(y_test=y_test, y_pred=y_pred,
                                    rf_results_path=ranking_results_path, dr_name=dr_name)
        joblib.dump(clf, rf_model_path + "/" + "random_forest" + "_" + dr_name + ".pkl")
        if self.compute_search_ranking:
            kt_list, pearson_coeff_list = self.classifier_ranking(clf=clf, test_data=x_test,
                                                                  av_train_dist=av_train_dist)
            kt_df = pd.DataFrame(kt_list)
            self.log.info(kt_df.describe())
            kt_df.to_msgpack(ranking_results_path + "/" + "kt_rf" + "_" + dr_name)

            pcc_df = pd.DataFrame(pearson_coeff_list)
            self.log.info(pcc_df.describe())
            pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_rf" + "_" + dr_name)

    def extra_trees(self, **kwargs):
        x_train = kwargs['x_train']
        y_train = kwargs['y_train']
        x_test = kwargs['x_test']
        y_test = kwargs['y_test']
        av_train_dist = kwargs['av_train_dist']
        dr_name = kwargs['dr_name']
        ranking_results_path = kwargs['ranking_results_path']
        et_model_path = kwargs['et_model_path']

        self.log.info("******* Extra Trees Classifier *******")
        clf = self.et_clf.classification(train_data=x_train, train_labels=y_train)
        y_pred = self.et_clf.prediction(clf=clf, test_data=x_test)
        self.et_clf.compute_metrics(y_test=y_test, y_pred=y_pred, et_results_path=ranking_results_path, dr_name=dr_name)
        joblib.dump(clf, et_model_path + "/" + "extra_trees" + "_" + dr_name + ".pkl")
        if self.compute_search_ranking:
            kt_list, pearson_coeff_list = self.classifier_ranking(clf=clf, test_data=x_test,
                                                                  av_train_dist=av_train_dist)
            kt_df = pd.DataFrame(kt_list)
            self.log.info(kt_df.describe())
            kt_df.to_msgpack(ranking_results_path + "/" + "kt_et" + "_" + dr_name)

            pcc_df = pd.DataFrame(pearson_coeff_list)
            self.log.info(pcc_df.describe())
            pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_et" + "_" + dr_name)

    def decision_tree(self, **kwargs):
        x_train = kwargs['x_train']
        y_train = kwargs['y_train']
        x_test = kwargs['x_test']
        y_test = kwargs['y_test']
        av_train_dist = kwargs['av_train_dist']
        dr_name = kwargs['dr_name']
        ranking_results_path = kwargs['ranking_results_path']
        dt_model_path = kwargs['dt_model_path']

        self.log.info("******* Decision Tree Classifier *******")
        clf = self.dt_clf.classification(train_data=x_train, train_labels=y_train)
        y_pred = self.dt_clf.prediction(clf=clf, test_data=x_test)
        self.dt_clf.compute_metrics(y_test=y_test, y_pred=y_pred,
                                    dt_results_path=ranking_results_path, dr_name=dr_name)
        joblib.dump(clf, dt_model_path + "/" + "decision_tree" + "_" + dr_name + ".pkl")
        if self.compute_search_ranking:
            kt_list, pearson_coeff_list = self.classifier_ranking(clf=clf, test_data=x_test,
                                                                  av_train_dist=av_train_dist)
            kt_df = pd.DataFrame(kt_list)
            self.log.info(kt_df.describe())
            kt_df.to_msgpack(ranking_results_path + "/" + "kt_dt" + "_" + dr_name)

            pcc_df = pd.DataFrame(pearson_coeff_list)
            self.log.info(pcc_df.describe())
            pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_dt" + "_" + dr_name)

    def adaboost(self, **kwargs):
        x_train = kwargs['x_train']
        y_train = kwargs['y_train']
        x_test = kwargs['x_test']
        y_test = kwargs['y_test']
        av_train_dist = kwargs['av_train_dist']
        dr_name = kwargs['dr_name']
        ranking_results_path = kwargs['ranking_results_path']
        adaboost_model_path = kwargs['adaboost_model_path']

        self.log.info("******* Adaboost Classifier *******")
        clf = self.adaboost_clf.classification(train_data=x_train, train_labels=y_train)
        y_pred = self.adaboost_clf.prediction(clf=clf, test_data=x_test)
        self.adaboost_clf.compute_metrics(y_test=y_test, y_pred=y_pred, dr_name=dr_name,
                                          adaboost_results_path=ranking_results_path)
        joblib.dump(clf, adaboost_model_path + "/" + "adaboost" + "_" + dr_name + ".pkl")
        if self.compute_search_ranking:
            kt_list, pearson_coeff_list = self.classifier_ranking(clf=clf, test_data=x_test,
                                                                  av_train_dist=av_train_dist)
            kt_df = pd.DataFrame(kt_list)
            self.log.info(kt_df.describe())
            kt_df.to_msgpack(ranking_results_path + "/" + "kt_adaboost" + "_" + dr_name)

            pcc_df = pd.DataFrame(pearson_coeff_list)
            self.log.info(pcc_df.describe())
            pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_adaboost" + "_" + dr_name)

    def sgd(self, **kwargs):
        x_train = kwargs['x_train']
        y_train = kwargs['y_train']
        x_test = kwargs['x_test']
        y_test = kwargs['y_test']
        av_train_dist = kwargs['av_train_dist']
        ranking_results_path = kwargs['ranking_results_path']
        dr_name = kwargs['dr_name']
        sgd_model_path = kwargs['sgd_model_path']

        self.log.info("******* Linear Classifier with SGD *******")
        clf = self.sgd_clf.classification(train_data=x_train, train_labels=y_train)
        y_pred = self.sgd_clf.prediction(clf=clf, test_data=x_test)
        self.sgd_clf.compute_metrics(y_test=y_test, y_pred=y_pred, dr_name=dr_name,
                                     sgd_results_path=ranking_results_path)
        joblib.dump(clf, sgd_model_path + "/" + "sgd" + "_" + dr_name + ".pkl")
        if self.compute_search_ranking:
            kt_list, pearson_coeff_list = self.classifier_ranking(clf=clf, test_data=x_test,
                                                                  av_train_dist=av_train_dist)
            kt_df = pd.DataFrame(kt_list)
            self.log.info(kt_df.describe())
            kt_df.to_msgpack(ranking_results_path + "/" + "kt_sgd" + "_" + dr_name)

            pcc_df = pd.DataFrame(pearson_coeff_list)
            self.log.info(pcc_df.describe())
            pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_sgd" + "_" + dr_name)

    def xgboost(self, **kwargs):
        x_train = kwargs['x_train']
        y_train = kwargs['y_train']
        x_test = kwargs['x_test']
        y_test = kwargs['y_test']
        av_train_dist = kwargs['av_train_dist']
        ranking_results_path = kwargs['ranking_results_path']
        dr_name = kwargs['dr_name']
        xgboost_model_path = kwargs['xgboost_model_path']

        self.log.info("******* XGBoost Classifier *******")
        clf = self.xgboost_clf.classification(train_data=x_train, train_labels=y_train)
        y_pred = self.xgboost_clf.prediction(clf=clf, test_data=x_test)
        self.xgboost_clf.compute_metrics(y_test=y_test, y_pred=y_pred, dr_name=dr_name,
                                         xgboost_results_path=ranking_results_path)
        joblib.dump(clf, xgboost_model_path + "/" + "xgboost" + dr_name + ".pkl")
        if self.compute_search_ranking:
            kt_list, pearson_coeff_list = self.classifier_ranking(clf=clf, test_data=x_test,
                                                                  av_train_dist=av_train_dist)
            kt_df = pd.DataFrame(kt_list)
            self.log.info(kt_df.describe())
            kt_df.to_msgpack(ranking_results_path + "/" + "kt_xgboost" + "_" + dr_name)

            pcc_df = pd.DataFrame(pearson_coeff_list)
            self.log.info(pcc_df.describe())
            pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_xgboost" + "_" + dr_name)

    def bagging(self, **kwargs):
        x_train = kwargs['x_train']
        y_train = kwargs['y_train']
        x_test = kwargs['x_test']
        y_test = kwargs['y_test']
        av_train_dist = kwargs['av_train_dist']
        ranking_results_path = kwargs['ranking_results_path']
        dr_name = kwargs['dr_name']
        bagging_model_path = kwargs['bagging_model_path']

        self.log.info("******* XGBoost Classifier *******")
        clf = self.bagging_clf.classification(train_data=x_train, train_labels=y_train)
        y_pred = self.bagging_clf.prediction(clf=clf, test_data=x_test)
        self.bagging_clf.compute_metrics(y_test=y_test, y_pred=y_pred, dr_name=dr_name,
                                         bagging_results_path=ranking_results_path)
        joblib.dump(clf, bagging_model_path + "/" + "bagging" + dr_name + ".pkl")
        if self.compute_search_ranking:
            kt_list, pearson_coeff_list = self.classifier_ranking(clf=clf, test_data=x_test,
                                                                  av_train_dist=av_train_dist)
            kt_df = pd.DataFrame(kt_list)
            self.log.info(kt_df.describe())
            kt_df.to_msgpack(ranking_results_path + "/" + "kt_bagging" + "_" + dr_name)

            pcc_df = pd.DataFrame(pearson_coeff_list)
            self.log.info(pcc_df.describe())
            pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_bagging" + "_" + dr_name)

    def train_classifiers(self, **kwargs):
        x_train = kwargs["x_train"]
        y_train = kwargs["y_train"]
        x_test = kwargs["x_test"]
        y_test = kwargs["y_test"]
        av_train_dist = kwargs["av_train_dist"]
        bnb_model_path = kwargs["bnb_model_path"]
        rf_model_path = kwargs["rf_model_path"]
        et_model_path = kwargs["et_model_path"]
        dt_model_path = kwargs["dt_model_path"]
        adaboost_model_path = kwargs["adaboost_model_path"]
        sgd_model_path = kwargs["sgd_model_path"]
        xgboost_model_path = kwargs["xgboost_model_path"]
        bagging_model_path = kwargs["bagging_model_path"]
        ranking_results_path = kwargs["ranking_results_path"]
        dr_name = kwargs["dr_name"]

        self.bernoulli_nb(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, av_train_dist=av_train_dist,
                          dr_name=dr_name, ranking_results_path=ranking_results_path, bnb_model_path=bnb_model_path)
        self.random_forest(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, av_train_dist=av_train_dist,
                           dr_name=dr_name, ranking_results_path=ranking_results_path, rf_model_path=rf_model_path)
        self.extra_trees(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, av_train_dist=av_train_dist,
                         dr_name=dr_name, ranking_results_path=ranking_results_path, et_model_path=et_model_path)
        self.decision_tree(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, av_train_dist=av_train_dist,
                           dr_name=dr_name, ranking_results_path=ranking_results_path, dt_model_path=dt_model_path)
        self.adaboost(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, av_train_dist=av_train_dist,
                      dr_name=dr_name, ranking_results_path=ranking_results_path,
                      adaboost_model_path=adaboost_model_path)
        self.sgd(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, av_train_dist=av_train_dist,
                 dr_name=dr_name, ranking_results_path=ranking_results_path, sgd_model_path=sgd_model_path)
        self.xgboost(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, av_train_dist=av_train_dist,
                     dr_name=dr_name, ranking_results_path=ranking_results_path, xgboost_model_path=xgboost_model_path)
        self.bagging(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, av_train_dist=av_train_dist,
                     dr_name=dr_name, ranking_results_path=ranking_results_path, bagging_model_path=bagging_model_path)

    def main(self):
        start_time = time()
        smote_path = self.config["data"]["smote_data"]
        ranking_results_path = self.config["data"]["ranking_results_path"]
        bnb_model_path = self.config["models"]["naive_bayes"]["model_path"]
        rf_model_path = self.config["models"]["random_forest"]["model_path"]
        et_model_path = self.config["models"]["extra_trees"]["model_path"]
        dt_model_path = self.config["models"]["decision_trees"]["model_path"]
        adaboost_model_path = self.config["models"]["adaboost"]["model_path"]
        sgd_model_path = self.config["models"]["sgd"]["model_path"]
        xgboost_model_path = self.config["models"]["xgboost"]["model_path"]
        bagging_model_path = self.config["models"]["bagging"]["model_path"]

        x_train, y_train, x_test, y_test, av_train_dist, av_test_dist = self.get_dr_matrices(smote_path=smote_path)
        self.train_classifiers(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                               av_train_dist=av_train_dist, bnb_model_path=bnb_model_path, rf_model_path=rf_model_path,
                               et_model_path=et_model_path, dt_model_path=dt_model_path,
                               adaboost_model_path=adaboost_model_path, sgd_model_path=sgd_model_path,
                               xgboost_model_path=xgboost_model_path, bagging_model_path=bagging_model_path,
                               ranking_results_path=ranking_results_path)
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    rm = RankingMetrics(use_pruned_data=True, compute_search_ranking=False)
    rm.main()
