from time import time

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB

from HelperFunctions.HelperFunction import HelperFunction
from HelperFunctions.LoadDRMatrices import LoadDRMatrices
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class CoreClassificationLogic:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    def perform_classification(self, train_data, train_labels, n_classes):
        start_time = time()
        chunk_size = 1000
        train_start_index = 0
        train_iter = 0
        clf = BernoulliNB()

        while train_start_index < train_data.shape[0]:
            if train_iter % chunk_size == 0:
                self.log.info("Iteration : #{}".format(train_iter))
            if train_start_index + chunk_size < train_data.shape[0]:
                p_matrix = train_data[train_start_index: train_start_index + chunk_size]
                p_labels = train_labels[train_start_index: train_start_index + chunk_size]
            else:
                p_matrix = train_data[train_start_index:]
                p_labels = train_labels[train_start_index:]
            if train_start_index == 0:
                clf.partial_fit(p_matrix, p_labels, n_classes)
            else:
                clf.partial_fit(p_matrix, p_labels)
            train_start_index += chunk_size
            train_iter += 1

        self.log.info("Completed fitting the model")
        self.log.info("Time taken : {}\n".format(time() - start_time))
        return clf

    def perform_prediction(self, test_data, classifier):
        start_time = time()
        test_start_index = 0
        test_iter = 0
        chunk_size = 1000
        test_pred = list()

        while test_start_index < test_data.shape[0]:
            if test_iter % chunk_size == 0:
                self.log.info("Iteration : #{}".format(test_iter))
            if test_start_index + chunk_size < test_data.shape[0]:
                p_matrix = test_data[test_start_index: test_start_index + chunk_size]
            else:
                p_matrix = test_data[test_start_index:]
            test_pred += [x for x in classifier.predict(p_matrix)]
            test_start_index += chunk_size
            test_iter += 1

        self.log.info("Completed predictions")
        self.log.info("Time taken : {}".format(time() - start_time))
        return test_pred

    def bernoulli_nb_classifier_top_k(self, clf, x_test, y_test, k):
        self.log.info("Predicting test data")

        start_time = time()
        test_start_index = 0
        test_iter = 0
        chunk_size = 1000
        top_k_acc_list = list()

        while test_start_index < x_test.shape[0]:
            if test_iter % chunk_size == 0:
                self.log.info("Iteration : #{}".format(test_iter))
            if test_start_index + chunk_size < x_test.shape[0]:
                p_matrix = x_test[test_start_index: test_start_index + chunk_size]
            else:
                p_matrix = x_test[test_start_index:]
            for index, pred_prob in enumerate(clf.predict_proba(p_matrix)):
                x, y = y_test[index], pred_prob.argsort()[-k:].tolist()
                if x in y:
                    top_k_acc = 1
                else:
                    top_k_acc = 0
                top_k_acc_list.append(top_k_acc)
            test_start_index += chunk_size
            test_iter += 1

        self.log.info("Completed predictions\nTime taken : {}".format(time() - start_time))
        return top_k_acc_list

    def bernoulli_nb_classifier_ranking(self, clf, x_test, av_train_dist):
        self.log.info("Predicting test data")

        start_time = time()
        test_start_index = 0
        test_iter = 0
        chunk_size = 1000
        kt_list = list()
        pearson_coeff_list = list()

        while test_start_index < x_test.shape[0]:
            if test_iter % chunk_size == 0:
                self.log.info("Iteration : #{}".format(test_iter))
            if test_start_index + chunk_size < x_test.shape[0]:
                p_matrix = x_test[test_start_index: test_start_index + chunk_size]
            else:
                p_matrix = x_test[test_start_index:]
            for index, pred_prob in enumerate(clf.predict_proba(p_matrix)):
                x, y = av_train_dist[index].toarray(), pred_prob
                kt = kendalltau(x, y)
                kt_list.append(kt)
                pcc = np.corrcoef(x, y)[:, 1]
                pearson_coeff_list.append(pcc)
            test_start_index += chunk_size
            test_iter += 1

        self.log.info("Completed predictions\nTime taken : {}".format(time() - start_time))
        return kt_list, pearson_coeff_list


class NaiveBayesClassifier:
    def __init__(self, use_pruned_data):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.helper = HelperFunction()
        self.load_dr_mat = LoadDRMatrices(use_pruned_data)
        self.use_pruned_data = use_pruned_data
        self.core_logic = CoreClassificationLogic()
        self.classifier_name = "naive_bayes"

    def compute_metrics(self, y_test, y_pred, dr_name, bnb_results_path):
        """

        :param y_test:
        :param y_pred:
        :param dr_name:
        :param bnb_results_path:
        :return:
        """
        results = dict()
        cr_report = classification_report(y_pred=y_pred, y_true=y_test)
        key_name = dr_name + "_" + str(self.classifier_name) + "_" + "cr_report"
        results[key_name] = cr_report
        cnf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)

        self.log.info("Classification Report : \n{}".format(cr_report))
        plt = self.helper.plot_cnf_matrix(cnf_matrix=cnf_matrix)
        plt_path = bnb_results_path + "/" + str(dr_name) + "_" + str(
            self.classifier_name) + "_" + "cnf_matrix" + ".png"
        self.log.info("Saving Confusion Matrix at path : {}".format(plt_path))
        plt.savefig(plt_path)
        return results

    def prediction(self, clf, test_data):
        """

        :param clf:
        :param test_data:
        :return:
        """
        self.log.info("Predicting on test data")
        y_pred = clf.predict(test_data)
        return y_pred

    def classification(self, train_data, train_labels):
        self.log.info("Using Naive Bayes classifier")
        cv = 5
        n_classes = np.unique(train_labels)
        clf = self.core_logic.perform_classification(train_data=train_data, train_labels=train_labels,
                                                     n_classes=n_classes)
        self.log.info("Performing cross validation with cv : {}".format(cv))
        scores = cross_val_score(clf, train_data, train_labels, cv=cv)
        self.log.info("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return clf

    def prepare_classifier(self, dr_matrices, labels, bnb_results_path):
        """

        :param dr_matrices:
        :param labels:
        :param bnb_results_path:
        :return:
        """
        dr_results_array = dict()
        for dr_name, dr_matrix in dr_matrices.items():
            if dr_name is "base_data":
                self.log.info("Using Random Forest classifier on Base Data")
            elif dr_name is "pca":
                self.log.info("Using Random Forest classifier on PCA")
            elif dr_name is "tsne_random":
                self.log.info("Using Random Forest classifier on TSNE with random init")
            elif dr_name is "tsne_pca":
                self.log.info("Using Random Forest classifier on TSNE with pca init")
            elif dr_name is "sae":
                self.log.info("Using Random Forest classifier on on SAE")
            else:
                self.log.error("Dimensionality Reduction technique employed is not supported!!!")
            x_train, x_test, y_train, y_test = self.helper.validation_split(dr_matrix, labels, test_size=0.25)
            del dr_matrix, labels
            clf = self.classification(train_data=x_train,
                                      train_labels=y_train)
            y_pred = self.prediction(clf=clf, test_data=x_test)
            results = self.compute_metrics(y_test=y_test, y_pred=y_pred, dr_name=dr_name,
                                           bnb_results_path=bnb_results_path)

            dr_results_array[dr_name] = results
        return dr_results_array

    def main(self, num_rows):
        start_time = time()
        labels_path = self.config["data"]["labels_path"]

        if self.use_pruned_data:
            base_data_path = self.config["data"]["pruned_feature_selection_path"]
        else:
            base_data_path = self.config["data"]["unpruned_feature_selection_path"]

        bnb_results_path = self.config["models"]["naive_bayes"]["results_path"]
        tsne_model_path = self.config["models"]["tsne"]["model_path"]
        sae_model_path = self.config["models"]["sae"]["model_path"]
        pca_model_path = self.config["models"]["pca"]["model_path"]

        dr_matrices, labels = self.load_dr_mat.get_dr_matrices(labels_path=labels_path, base_data_path=base_data_path,
                                                               pca_model_path=pca_model_path,
                                                               tsne_model_path=tsne_model_path,
                                                               sae_model_path=sae_model_path, num_rows=num_rows)
        self.prepare_classifier(dr_matrices=dr_matrices, labels=labels, bnb_results_path=bnb_results_path)
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    bnb_clf = NaiveBayesClassifier(use_pruned_data=True)
    bnb_clf.main(num_rows=50000)
