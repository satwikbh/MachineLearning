import numpy as np

from time import time

from sklearn.naive_bayes import BernoulliNB
from scipy.stats import kendalltau

from Utils.LoggerUtil import LoggerUtil


class NaiveBayesClassifier:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    def perform_classification(self, x, y, n_classes):
        clf = BernoulliNB()

        start_time = time()
        train_start_index = 0
        train_iter = 0

        while train_start_index < x.shape[0]:
            if train_iter % 100 == 0:
                self.log.info("Iteration : #{}".format(train_iter))
            if train_start_index + 1000 > x.shape[0]:
                p_matrix = x[train_start_index:]
                p_labels = y[train_start_index:]
            else:
                p_matrix = x[train_start_index: train_start_index + 1000]
                p_labels = y[train_start_index: train_start_index + 1000]
            if train_start_index == 0:
                clf.partial_fit(p_matrix, p_labels, n_classes)
            else:
                clf.partial_fit(p_matrix, p_labels)
            train_start_index += 1000
            train_iter += 1

        self.log.info("Completed fitting the model")
        self.log.info("Time taken : {}\n".format(time() - start_time))
        return clf

    def perform_prediction(self, x, classifier):
        start_time = time()
        test_start_index = 0
        test_iter = 0
        test_pred = list()

        while test_start_index < x.shape[0]:
            if test_iter % 100 == 0:
                self.log.info("Iteration : #{}".format(test_iter))
            if test_start_index + 1000 > x.shape[0]:
                p_matrix = x[test_start_index:]
            else:
                p_matrix = x[test_start_index: test_start_index + 1000]
            test_pred += [x for x in classifier.predict(p_matrix)]
            test_start_index += 1000
            test_iter += 1

        self.log.info("Completed predictions")
        self.log.info("Time taken : {}".format(time() - start_time))
        return test_pred

    def bernouille_nb_classifier_top_k(self, clf, x_test, y_test, k):
        self.log.info("Predicting test data")

        start_time = time()
        test_start_index = 0
        test_iter = 0
        top_k_acc_list = list()

        while test_start_index < x_test.shape[0]:
            if test_iter % 100 == 0:
                self.log.info("Iteration : #{}".format(test_iter))
            if test_start_index + 1000 > x_test.shape[0]:
                p_matrix = x_test[test_start_index:]
            else:
                p_matrix = x_test[test_start_index: test_start_index + 1000]
            for index, pred_prob in enumerate(clf.predict_proba(p_matrix)):
                x, y = y_test[index], pred_prob.argsort()[-k:].tolist()
                if x in y:
                    top_k_acc = (y.index(x) * 1.0) / len(y)
                else:
                    top_k_acc = 0.0
                top_k_acc_list.append(top_k_acc)
            test_start_index += 1000
            test_iter += 1

        self.log.info("Completed predictions\nTime taken : {}".format(time() - start_time))
        return top_k_acc_list

    def bernouille_nb_classifier_ranking(self, clf, x_test, av_test_dist):
        self.log.info("Predicting test data")

        start_time = time()
        chunk_size = 1000
        test_start_index = 0
        test_iter = 0
        kt_list = list()
        pearson_coeff_list = list()

        while test_start_index < x_test.shape[0]:
            if test_iter % chunk_size == 0:
                self.log.info("Iteration : #{}".format(test_iter))
            if test_start_index + chunk_size > x_test.shape[0]:
                p_matrix = x_test[test_start_index:]
            else:
                p_matrix = x_test[test_start_index: test_start_index + chunk_size]
            for index, pred_prob in enumerate(clf.predict_proba(p_matrix)):
                x, y = av_test_dist[index].toarray(), pred_prob
                kt = kendalltau(x, y)
                kt_list.append(kt)
                pcc = np.corrcoef(x, y)[:, 1]
                pearson_coeff_list.append(pcc)
            test_start_index += chunk_size
            test_iter += 1

        self.log.info("Completed predictions\nTime taken : {}".format(time() - start_time))
        return kt_list, pearson_coeff_list
