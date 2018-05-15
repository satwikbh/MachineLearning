from sklearn.naive_bayes import BernoulliNB
from time import time

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
