import numpy as np
import glob

from time import time

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from PrepareData.LoadData import LoadData
from HelperFunctions.HelperFunction import HelperFunction


class OnlinePipeline(Pipeline):
    def partial_fit(self, x, y=None):
        for i, step in enumerate(self.steps):
            name, est = step
            est.partial_fit(x, y)
        return self


class SVMGridSearch(object):
    """
    Performs GridSearch to find the best params for SVM
    """

    def __init__(self, multi_class, large_dataset):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.load_data = LoadData()
        self.config = ConfigUtil.get_config_instance()
        self.helper = HelperFunction()
        self.scores = ['precision', 'recall']
        self.large_dataset = large_dataset
        self.multi_class = multi_class

    def tuned_parameters(self):
        if self.large_dataset:
            tuned_params = [
                {
                    'ovr__estimator__warm_start': [True],
                    'ovr__estimator__n_jobs': [30],
                    'ovr__estimator__alpha': 10.0 ** -np.arange(1, 7),
                    'ovr__estimator__max_iter': (10, 50, 80),
                    'ovr__estimator__penalty': ('l2', 'elasticnet')
                }
            ]
        else:
            tuned_params = [
                {'estimator__kernel': ['rbf'], 'estimator__gamma': [1e-3, 1e-4], 'estimator__C': [1, 10, 100, 1000]},
                {'estimator__kernel': ['linear'], 'estimator__C': [1, 10, 100, 1000]}
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

    def perform_grid_search(self, tuned_parameters, input_matrix, dr_name, labels, svm_results_path):
        results = dict()
        x_train, x_test, y_train, y_test = self.helper.validation_split(input_matrix, labels, test_size=0.25)
        if self.large_dataset:
            pipe = OnlinePipeline([('ovr', OneVsRestClassifier(SGDClassifier()))])
            clf = GridSearchCV(pipe, tuned_parameters, cv=5)
        else:
            clf = GridSearchCV(OneVsRestClassifier(SVC()), tuned_parameters, cv=5, n_jobs=30)
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
        try:
            if not self.multi_class:
                auroc_score = roc_auc_score(y_true=y_true, y_score=y_pred)
                results['auroc'] = auroc_score
                self.log.info(auroc_score)
            cr_report = classification_report(y_true, y_pred)
            results['cr_report'] = cr_report
            self.log.info(cr_report)

            cnf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
            if cnf_matrix.shape[0] != cnf_matrix.shape[1]:
                raise Exception
            plt = self.helper.plot_cnf_matrix(cnf_matrix=cnf_matrix)
            plt.savefig(svm_results_path + "/" + "cnf_matrix_" + str(dr_name) + ".png")
            return results
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def prepare_gridsearch(self, dr_matrices, tuned_parameters, labels, svm_results_path):
        dr_results_array = dict()
        for dr_name, dr_matrix in dr_matrices.items():
            if dr_name is "base_data":
                self.log.info("Performing GridSearch for SVM on Base Data")
            elif dr_name is "pca":
                self.log.info("Performing GridSearch for SVM on PCA")
            elif dr_name is "tsne_random":
                self.log.info("Performing GridSearch for SVM on TSNE with random init")
            elif dr_name is "tsne_pca":
                self.log.info("Performing GridSearch for SVM on TSNE with pca init")
            elif dr_name is "sae":
                self.log.info("Performing GridSearch for SVM on on SAE")
            else:
                self.log.error("Dimensionality Reduction technique employed is not supported!!!")
            dr_results_array[dr_name] = self.perform_grid_search(tuned_parameters=tuned_parameters,
                                                                 input_matrix=dr_matrix,
                                                                 dr_name=dr_name,
                                                                 labels=labels,
                                                                 svm_results_path=svm_results_path)
        return dr_results_array

    def main(self, num_rows):
        start_time = time()
        self.log.info("GridSearch on SVM started")

        labels_path = self.config["data"]["labels_path"]
        base_data_path = self.config["data"]["pruned_feature_vector_path"]
        pca_model_path = self.config["models"]["pca"]["model_path"]
        tsne_model_path = self.config["models"]["tsne"]["model_path"]
        sae_model_path = self.config["models"]["sae"]["model_path"]
        svm_results_path = self.config["models"]["svm"]["results_path"]

        tuned_params = self.tuned_parameters()
        dr_matrices, labels = self.get_dr_matrices(labels_path=labels_path, base_data_path=base_data_path,
                                                   pca_model_path=pca_model_path, tsne_model_path=tsne_model_path,
                                                   sae_model_path=sae_model_path, num_rows=num_rows)
        dr_results_array = self.prepare_gridsearch(dr_matrices=dr_matrices, tuned_parameters=tuned_params,
                                                   labels=labels, svm_results_path=svm_results_path)
        np.savetxt(fname=svm_results_path + "/" + "svm_gridsearch", X=dr_results_array)
        self.log.info("GridSearch on SVM completed")
        self.log.info("Time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    svm_grid_search = SVMGridSearch(multi_class=True, large_dataset=True)
    svm_grid_search.main(num_rows=50000)