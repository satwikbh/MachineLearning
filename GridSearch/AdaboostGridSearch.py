import glob
from time import time

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from HelperFunctions.HelperFunction import HelperFunction
from PrepareData.LoadData import LoadData
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class AdaboostGridSearch:
    def __init__(self, multi_class):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()
        self.helper = HelperFunction()
        self.multi_class = multi_class

    @staticmethod
    def tuned_parameters():
        tuned_params = [
            {'estimator__n_estimators': range(10, 100, 10)}
        ]
        return tuned_params

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

    @staticmethod
    def validation_split(input_matrix, labels, test_size):
        x_train, x_test, y_train, y_test = train_test_split(input_matrix, labels, test_size=test_size, random_state=0)
        return x_train, x_test, y_train, y_test

    def perform_grid_search(self, tuned_parameters, dr_name, input_matrix, labels, adaboost_results_path, call=False):
        """

        :param tuned_parameters:
        :param dr_name:
        :param input_matrix:
        :param labels:
        :param adaboost_results_path:
        :param call: This means that the function is being called from outside which needs only the classifier object.
        :return:
        """
        results = dict()
        x_train, x_test, y_train, y_test = self.validation_split(input_matrix, labels, test_size=0.25)
        clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=5, n_jobs=30)
        if hasattr(x_train, "toarray"):
            clf.fit(x_train.toarray(), y_train)
        else:
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
        if call:
            return clf
        y_true, y_pred = y_test, clf.predict(x_test)
        try:
            if not self.multi_class:
                auroc_score = roc_auc_score(y_true=y_true, y_score=y_pred)
                results['auroc_score'] = auroc_score
                self.log.info(auroc_score)
            cr_report = classification_report(y_true, y_pred)
            results['cr_report'] = cr_report
            self.log.info(cr_report)

            cnf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
            if cnf_matrix.shape[0] != cnf_matrix.shape[1]:
                raise Exception
            plt = self.helper.plot_cnf_matrix(cnf_matrix=cnf_matrix)
            plt.savefig(adaboost_results_path + "/" + "cnf_matrix_" + str(dr_name) + ".png")
            return results
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def prepare_gridsearch(self, dr_matrices, tuned_parameters, labels, adaboost_results_path):
        dr_results_array = dict()
        for dr_name, dr_matrix in dr_matrices.items():
            if dr_name is "base_data":
                self.log.info("Performing GridSearch for Adaboost on Base Data")
            elif dr_name is "pca":
                self.log.info("Performing GridSearch for Adaboost on PCA")
            elif dr_name is "tsne_random":
                self.log.info("Performing GridSearch for Adaboost on TSNE with random init")
            elif dr_name is "tsne_pca":
                self.log.info("Performing GridSearch for Adaboost on TSNE with pca init")
            elif dr_name is "sae":
                self.log.info("Performing GridSearch for Adaboost on on SAE")
            else:
                self.log.error("Dimensionality Reduction technique employed is not supported!!!")
            dr_results_array[dr_name] = self.perform_grid_search(tuned_parameters=tuned_parameters,
                                                                 input_matrix=dr_matrix,
                                                                 dr_name=dr_name,
                                                                 labels=labels,
                                                                 adaboost_results_path=adaboost_results_path,
                                                                 call=True)
        return dr_results_array

    def main(self, num_rows):
        start_time = time()
        self.log.info("GridSearch on Adaboost started")

        labels_path = self.config["data"]["labels_path"]
        base_data_path = self.config["data"]["pruned_feature_vector_path"]
        pca_model_path = self.config["models"]["pca"]["model_path"]
        tsne_model_path = self.config["models"]["tsne"]["model_path"]
        sae_model_path = self.config["models"]["sae"]["model_path"]
        adaboost_results_path = self.config["models"]["adaboost"]["results_path"]

        tuned_params = self.tuned_parameters()
        dr_matrices, labels = self.get_dr_matrices(labels_path=labels_path, base_data_path=base_data_path,
                                                   pca_model_path=pca_model_path, tsne_model_path=tsne_model_path,
                                                   sae_model_path=sae_model_path, num_rows=num_rows)
        dr_results_array = self.prepare_gridsearch(dr_matrices=dr_matrices, tuned_parameters=tuned_params,
                                                   labels=labels, adaboost_results_path=adaboost_results_path)
        dr_results_df = pd.DataFrame(dr_results_array)
        dr_results_df.to_msgpack(adaboost_results_path + "/" + "adaboost_gridsearch")
        self.log.info("GridSearch on Random Forest completed")
        self.log.info("Time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    adaboost_gs = AdaboostGridSearch(multi_class=True)
    adaboost_gs.main(num_rows=50000)
