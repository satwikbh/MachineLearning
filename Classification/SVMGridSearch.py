import pickle as pi

from time import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from PrepareData.LoadData import LoadData


class SVMGridSearch(object):
    """
    Performs GridSearch to find the best params for SVM
    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.load_data = LoadData()
        self.config = ConfigUtil.get_config_instance()
        self.scores = ['precision', 'recall']

    def get_data(self, num_rows):
        labels_path = self.config['data']['labels_path']
        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        labels = pi.load(open(labels_path + "labels.pkl"))
        return input_matrix, labels

    @staticmethod
    def tuned_parameters():
        tuned_params = [
            {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
        ]
        return tuned_params

    @staticmethod
    def validation_split(input_matrix, labels, test_size):
        x_train, x_test, y_train, y_test = train_test_split(input_matrix, labels, test_size=test_size, random_state=0)
        return x_train, x_test, y_train, y_test

    def perform_grid_search(self, tuned_parameters, x_train, x_test, y_train, y_test):
        for score in self.scores:
            clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
            clf.fit(x_train, y_train)
            best_params = clf.best_params_
            self.log.info("Best parameters set found on development set : \n{}".format(best_params))
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                self.log.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            self.log.info("Detailed classification report:")
            self.log.info("The model is trained on the full development set.")
            self.log.info("The scores are computed on the full evaluation set.")
            y_true, y_pred = y_test, clf.predict(x_test)
            self.log.info(classification_report(y_true, y_pred))

    def main(self, num_rows):
        start_time = time()
        self.log.info("GridSearch on SVM started")
        tuned_params = self.tuned_parameters()
        input_matrix, labels = self.get_data(num_rows=num_rows)
        x_train, x_test, y_train, y_test = self.validation_split(input_matrix, labels, test_size=0.25)
        self.perform_grid_search(tuned_parameters=tuned_params,
                                 x_train=x_train, x_test=x_test,
                                 y_train=y_train, y_test=y_test)
        self.log.info("GridSearch on SVM completed")
        self.log.info("Time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    svm_grid_search = SVMGridSearch()
    svm_grid_search.main(num_rows=50000)
