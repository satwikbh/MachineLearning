import pandas as pd
import numpy as np

from time import time

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import kendalltau
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from NaiveBayesClassifier import NaiveBayesClassifier
from RandomForestGridSearch import RandomForestGridSearch
from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from HelperFunctions.HelperFunction import HelperFunction


class RankingMetrics:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.helper = HelperFunction()
        self.rf_gridsearch = RandomForestGridSearch(multi_class=False)

    @staticmethod
    def classifier_ranking(clf, data, av_test_dist):
        kt_list = list()
        pearson_coeff_list = list()
        for index, pred_prob in enumerate(clf.predict_proba(data)):
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
                     n_classes, plot_path, ranking_results_path):
        self.log.info("******* Bernouille Naive Bayes Classifier *******")
        bnb_clf = NaiveBayesClassifier()
        classifier = bnb_clf.perform_classification(train_data=x_train_smote, train_labels=y_train_smote,
                                                    n_classes=n_classes)
        test_preds = bnb_clf.perform_prediction(test_data=x_test_smote, classifier=classifier)
        acc_score = accuracy_score(y_pred=test_preds, y_true=y_test_smote)
        self.log.info("Accuracy Score : {}".format(acc_score))
        cr_report = classification_report(y_pred=test_preds, y_true=y_test_smote)
        self.log.info("Classification Report : \n{}".format(cr_report))
        cnf_matrix = confusion_matrix(y_pred=test_preds, y_true=y_test_smote)
        plt = self.helper.plot_cnf_matrix(cnf_matrix)
        plt.savefig(plot_path + "/" + "bnb_smote_data.png")

        kt_list, pearson_coeff_list = bnb_clf.bernouille_nb_classifier_ranking(clf=classifier, x_test=x_test_smote,
                                                                               av_test_dist=av_test_dist)

        kt_df = pd.DataFrame(kt_list)
        self.log.info(kt_df.describe())
        kt_df.to_msgpack(ranking_results_path + "/" + "kt_bnb_smote")

        pcc_df = pd.DataFrame(pearson_coeff_list)
        self.log.info(pcc_df.describe())
        pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_bnb_smote")

    def random_forest(self, x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, plot_path,
                      ranking_results_path):
        self.log.info("******* Random Forest Classifier *******")
        start_time = time()
        rf_clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=100, oob_score=False, n_jobs=30)
        rf_clf.fit(x_train_smote, y_train_smote)
        scores = cross_val_score(rf_clf, x_train_smote, y_train_smote, cv=5)
        self.log.info("RandomForest Classifier\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        self.log.info("Time taken : {}".format(time() - start_time))

        y_pred = rf_clf.predict(X=x_test_smote)
        cr_report = classification_report(y_pred=y_pred, y_true=y_test_smote)
        self.log.info("Classification Report : \n{}".format(cr_report))

        cnf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test_smote)
        plt = self.helper.plot_cnf_matrix(cnf_matrix=cnf_matrix)
        plt.savefig(plot_path + "/" + "rf_smote_data.png")

        kt_list, pearson_coeff_list = self.classifier_ranking(clf=rf_clf, data=x_test_smote, av_test_dist=av_test_dist)

        kt_df = pd.DataFrame(kt_list)
        self.log.info(kt_df.describe())
        kt_df.to_msgpack(ranking_results_path + "/" + "kt_rf_smote")

        pcc_df = pd.DataFrame(pearson_coeff_list)
        self.log.info(pcc_df.describe())
        pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_rf_smote")

    def extra_trees(self, x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, plot_path,
                    ranking_results_path):
        """
        Since Extra Trees Classifier doesn't have a seperate class, including the entire code here.
        # TODO : Need to separate as in Random Forest Method.
        :param x_train_smote:
        :param y_train_smote:
        :param x_test_smote:
        :param y_test_smote:
        :param av_test_dist:
        :param plot_path:
        :param ranking_results_path:
        :return:
        """
        self.log.info("******* Extra Trees Classifier *******")
        start_time = time()
        et_clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
        scores = cross_val_score(et_clf, x_train_smote, y_train_smote, cv=5)
        self.log.info("ExtraTreesClassifier\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        self.log.info("Time taken : {}".format(time() - start_time))

        et_clf.fit(X=x_train_smote, y=y_train_smote)
        y_pred = et_clf.predict(X=x_test_smote)

        cr_report = classification_report(y_pred=y_pred, y_true=y_test_smote)
        self.log.info("Classification Report : \n{}".format(cr_report))
        cnf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test_smote)
        plt = self.helper.plot_cnf_matrix(cnf_matrix=cnf_matrix)
        plt.savefig(plot_path + "/" + "et_smote_data.png")

        kt_list, pearson_coeff_list = self.classifier_ranking(clf=et_clf, data=x_test_smote, av_test_dist=av_test_dist)

        kt_df = pd.DataFrame(kt_list)
        self.log.info(kt_df.describe())
        kt_df.to_msgpack(ranking_results_path + "/" + "kt_et_smote")

        pcc_df = pd.DataFrame(pearson_coeff_list)
        self.log.info(pcc_df.describe())
        pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_et_smote")

    def decision_tree(self, x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, plot_path,
                      ranking_results_path):
        """
        Since Decision Tree Classifier doesn't have a separate class, including the entire code here.
        # TODO : Need to separate as in Random Forest Method.
        :param x_train_smote:
        :param y_train_smote:
        :param x_test_smote:
        :param y_test_smote:
        :param av_test_dist:
        :param plot_path:
        :param ranking_results_path:
        :return:
        """
        self.log.info("******* Decision Tree Classifier *******")
        start_time = time()
        dt_clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
        scores = cross_val_score(dt_clf, x_train_smote, y_train_smote, cv=5)
        self.log.info("Decision Tree Classifier\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        self.log.info("Time taken : {}".format(time() - start_time))

        dt_clf.fit(X=x_train_smote, y=y_train_smote)
        y_pred = dt_clf.predict(X=x_test_smote)

        cr_report = classification_report(y_pred=y_pred, y_true=y_test_smote)
        self.log.info("Classification Report : \n{}".format(cr_report))
        cnf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test_smote)
        plt = self.helper.plot_cnf_matrix(cnf_matrix=cnf_matrix)
        plt.savefig(plot_path + "/" + "dt_smote_data.png")

        kt_list, pearson_coeff_list = self.classifier_ranking(clf=dt_clf, data=x_test_smote, av_test_dist=av_test_dist)

        kt_df = pd.DataFrame(kt_list)
        self.log.info(kt_df.describe())
        kt_df.to_msgpack(ranking_results_path + "/" + "kt_dt_smote")

        pcc_df = pd.DataFrame(pearson_coeff_list)
        self.log.info(pcc_df.describe())
        pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_dt_smote")

    def adaboost(self, x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, plot_path,
                 ranking_results_path):
        self.log.info("******* Decision Tree Classifier *******")
        start_time = time()
        adaboost_clf = AdaBoostClassifier(n_estimators=100)
        scores = cross_val_score(adaboost_clf, x_train_smote, y_train_smote, cv=5)
        self.log.info("Adaboost Classifier\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        self.log.info("Time taken : {}".format(time() - start_time))

        adaboost_clf.fit(X=x_train_smote, y=y_train_smote)
        y_pred = adaboost_clf.predict(X=x_test_smote)

        cr_report = classification_report(y_pred=y_pred, y_true=y_test_smote)
        self.log.info("Classification Report : \n{}".format(cr_report))
        cnf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test_smote)
        plt = self.helper.plot_cnf_matrix(cnf_matrix=cnf_matrix)
        plt.savefig(plot_path + "/" + "adaboost_smote_data.png")

        kt_list, pearson_coeff_list = self.classifier_ranking(clf=adaboost_clf, data=x_test_smote,
                                                              av_test_dist=av_test_dist)

        kt_df = pd.DataFrame(kt_list)
        self.log.info(kt_df.describe())
        kt_df.to_msgpack(ranking_results_path + "/" + "kt_adaboost_smote")

        pcc_df = pd.DataFrame(pearson_coeff_list)
        self.log.info(pcc_df.describe())
        pcc_df.to_msgpack(ranking_results_path + "/" + "pcc_adaboost_smote")

    def train_classifiers(self, x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, n_classes,
                          plot_path, ranking_results_path):
        self.bernoulli_nb(x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, n_classes, plot_path,
                          ranking_results_path)
        self.random_forest(x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, plot_path,
                           ranking_results_path)
        self.extra_trees(x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, plot_path,
                         ranking_results_path)
        self.decision_tree(x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, plot_path,
                           ranking_results_path)
        self.adaboost(x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, plot_path,
                      ranking_results_path)

    def main(self):
        start_time = time()
        plot_path = self.config["plots"]["smote"]
        smote_path = self.config["data"]["smote_data"]
        ranking_results_path = self.config["data"]["ranking_results_path"]

        x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_train_dist, av_test_dist = self.get_dr_matrices(
            smote_path=smote_path)
        n_classes = np.unique(y_train_smote)
        self.train_classifiers(x_train_smote, y_train_smote, x_test_smote, y_test_smote, av_test_dist, n_classes,
                               plot_path, ranking_results_path)
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    rm = RankingMetrics()
    rm.main()
