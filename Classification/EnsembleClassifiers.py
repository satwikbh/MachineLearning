import numpy as np
import glob
import json

from time import time
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from NaiveBayesClassifier import NaiveBayesClassifier
from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from PrepareData.LoadData import LoadData
from HelperFunctions.HelperFunction import HelperFunction


class EnsembleClassifiers:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()
        self.helper = HelperFunction()
        self.naive_bayes = NaiveBayesClassifier()
        self.classifiers_list = ['adaboost', 'random_forest', 'extra_trees', 'decision_trees']

    def get_dr_matrices(self, labels_path, base_data_path, pca_model_path, tsne_model_path, sae_model_path, num_rows):
        """
        Takes the dimensionality reduction techniques model_path's, loads the matrices.
        Returns the matrices as a dict.
        :param labels_path:
        :param base_data_path:
        :param pca_model_path:
        :param tsne_model_path:
        :param sae_model_path:
        :param num_rows:
        :return:
        """
        dr_matrices = dict()

        input_matrix, input_matrix_indices, labels = self.load_data.get_data_with_labels(num_rows=num_rows,
                                                                                         data_path=base_data_path,
                                                                                         labels_path=labels_path)
        del input_matrix_indices
        dr_matrices['base_data'] = input_matrix

        pca_file_name = pca_model_path + "/" + "pca_reduced_matrix_" + str(num_rows) + ".npy"
        pca_reduced_matrix = np.load(pca_file_name)
        dr_matrices["pca"] = pca_reduced_matrix

        tsne_random_file_name = glob.glob(tsne_model_path + "/" + "tsne_reduced_matrix_init_random_*")[0]
        tsne_random_reduced_matrix = np.load(tsne_random_file_name)['arr']
        dr_matrices["tsne_random"] = tsne_random_reduced_matrix

        tsne_pca_file_name = glob.glob(tsne_model_path + "/" + "tsne_reduced_matrix_init_pca_*")[0]
        tsne_pca_reduced_matrix = np.load(tsne_pca_file_name)['arr']
        dr_matrices["tsne_pca"] = tsne_pca_reduced_matrix

        sae_file_name = sae_model_path + "/" + "sae_reduced_matrix_" + str(num_rows) + ".npz"
        sae_reduced_matrix = np.load(sae_file_name)['arr_0']
        dr_matrices['sae'] = sae_reduced_matrix

        return dr_matrices, labels

    def perform_classification(self, input_matrix, dr_name, labels, ensemble_results_path):
        """
        Why is the method so complicated?
        # TODO : Split each classifier into separate classes.
        :param input_matrix:
        :param dr_name:
        :param labels:
        :param ensemble_results_path:
        :return:
        """
        results = dict()
        cv = 5
        n_classes = len(np.unique(labels))
        x_train, x_test, y_train, y_test = self.helper.validation_split(input_matrix, labels, test_size=0.25)
        del input_matrix, labels
        for classifier in self.classifiers_list:
            start_time = time()
            if classifier is 'naive_bayes':
                self.log.info("Using Bernouille Naive Bayes classifier")
                clf = self.naive_bayes.perform_classification(train_data=x_train, train_labels=y_train,
                                                              n_classes=n_classes)
            elif classifier is 'adaboost':
                self.log.info("Using Adaboost classifier")
                clf = AdaBoostClassifier(n_estimators=100, random_state=0)
            elif classifier is 'random_forest':
                self.log.info("Using Random Forest classifier")
                clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=0)
            elif classifier is 'extra_trees':
                self.log.info("Using Extra Trees classifier")
                clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
            elif classifier is 'decision_trees':
                self.log.info("Using Decision Trees classifier")
                clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
            else:
                self.log.error("Pick from adaboost, random_forest, extra_trees, decision_trees. Others not supported")
                clf = None
            self.log.info("Performing cross validation with cv : {}".format(cv))
            scores = cross_val_score(clf, x_train, y_train, cv=cv)
            self.log.info("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

            if classifier is 'naive_bayes':
                y_pred = self.naive_bayes.perform_prediction(test_data=x_test, classifier=clf)
            else:
                clf.fit(X=x_train, y=y_train)
                y_pred = clf.predict(X=x_test)

            cr_report = classification_report(y_pred=y_pred, y_true=y_test)
            key_name = dr_name + "_" + classifier + "_" + "cr_report"
            results[key_name] = cr_report
            cnf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)

            self.log.info("Classification Report : \n{}".format(cr_report))
            plt = self.helper.plot_cnf_matrix(cnf_matrix=cnf_matrix)
            plt_path = ensemble_results_path + "/" + str(dr_name) + "_" + str(classifier) + "_" + "cnf_matrix" + ".png"
            self.log.info("Saving Confusion Matrix at path : {}".format(plt_path))
            plt.savefig(plt_path)
            self.log.info("Time taken : {}".format(time() - start_time))
        return results

    def prepare_classifier(self, dr_matrices, labels, ensemble_results_path):
        dr_results_array = dict()
        for dr_name, dr_matrix in dr_matrices.items():
            if dr_name is "base_data":
                self.log.info("Using Ensemble classifiers on Base Data")
            elif dr_name is "pca":
                self.log.info("Using Ensemble classifiers on PCA")
            elif dr_name is "tsne_random":
                self.log.info("Using Ensemble classifiers on TSNE with random init")
            elif dr_name is "tsne_pca":
                self.log.info("Using Ensemble classifiers on TSNE with pca init")
            elif dr_name is "sae":
                self.log.info("Using Ensemble classifiers on on SAE")
            else:
                self.log.error("Dimensionality Reduction technique employed is not supported!!!")
            dr_results_array[dr_name] = self.perform_classification(input_matrix=dr_matrix,
                                                                    dr_name=dr_name,
                                                                    labels=labels,
                                                                    ensemble_results_path=ensemble_results_path)
        return dr_results_array

    def main(self, num_rows):
        start_time = time()
        self.log.info("Using Ensemble of Classifiers")

        labels_path = self.config["data"]["labels_path"]
        base_data_path = self.config["data"]["feature_selection_path"]
        pca_model_path = self.config["models"]["pca"]["model_path"]
        tsne_model_path = self.config["models"]["tsne"]["model_path"]
        sae_model_path = self.config["models"]["sae"]["model_path"]
        ensemble_results_path = self.config["models"]["ensemble"]["results_path"]

        dr_matrices, labels = self.get_dr_matrices(labels_path=labels_path, base_data_path=base_data_path,
                                                   pca_model_path=pca_model_path, tsne_model_path=tsne_model_path,
                                                   sae_model_path=sae_model_path, num_rows=num_rows)
        dr_results_array = self.prepare_classifier(dr_matrices=dr_matrices, labels=labels,
                                                   ensemble_results_path=ensemble_results_path)
        file_name = ensemble_results_path + "/" + "ensemble_classifiers_results.json"
        json.dump(dr_results_array, file_name)
        self.log.info("Classification completed")
        self.log.info("Time taken : {}".format(time() - start_time))


if __name__ == "__main__":
    ensemble = EnsembleClassifiers()
    ensemble.main(num_rows=25000)
