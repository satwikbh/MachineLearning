from time import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

from HelperFunctions.HelperFunction import HelperFunction
from HelperFunctions.LoadDRMatrices import LoadDRMatrices
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class RandomForestGridSearch:
    """
    Performs GridSearch to find best params for Random Forest.
    """

    def __init__(self, multi_class, use_pruned_data):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.helper = HelperFunction()
        self.load_data = LoadDRMatrices(use_pruned_data=use_pruned_data)
        self.multi_class = multi_class
        self.use_pruned_data = use_pruned_data

    @staticmethod
    def tuned_parameters():
        tuned_params = [
            {'estimator__n_estimators': range(10, 100, 10), 'estimator__min_samples_leaf': range(50, 100, 25),
             'estimator__random_state': [11], 'estimator__oob_score': [False]}
        ]
        return tuned_params

    def perform_grid_search(self, tuned_parameters, dr_name, input_matrix, labels, rf_results_path, call=False):
        """

        :param tuned_parameters:
        :param dr_name:
        :param input_matrix:
        :param labels:
        :param rf_results_path:
        :param call: This means that the function is being called from outside which needs only the classifier object.
        :return:
        """
        results = dict()
        x_train, x_test, y_train, y_test = self.helper.validation_split(input_matrix, labels, test_size=0.25)
        clf = GridSearchCV(OneVsRestClassifier(RandomForestClassifier()), tuned_parameters, cv=5, n_jobs=30)
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
            plt.savefig(rf_results_path + "/" + "cnf_matrix_gs_" + str(dr_name) + ".png")
            return results
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def prepare_gridsearch(self, dr_matrices, tuned_parameters, labels, rf_results_path):
        dr_results_array = dict()
        for dr_name, dr_matrix in dr_matrices.items():
            if dr_name is "base_data":
                self.log.info("Performing GridSearch for Random Forest on Base Data")
            elif dr_name is "pca":
                self.log.info("Performing GridSearch for Random Forest on PCA")
            elif dr_name is "tsne_random":
                self.log.info("Performing GridSearch for Random Forest on TSNE with random init")
            elif dr_name is "tsne_pca":
                self.log.info("Performing GridSearch for Random Forest on TSNE with pca init")
            elif dr_name is "sae":
                self.log.info("Performing GridSearch for Random Forest on on SAE")
            else:
                self.log.error("Dimensionality Reduction technique employed is not supported!!!")
            dr_results_array[dr_name] = self.perform_grid_search(tuned_parameters=tuned_parameters,
                                                                 input_matrix=dr_matrix,
                                                                 dr_name=dr_name,
                                                                 labels=labels,
                                                                 rf_results_path=rf_results_path)
        return dr_results_array

    def main(self, num_rows):
        start_time = time()
        self.log.info("GridSearch on Random Forest started")

        labels_path = self.config["data"]["labels_path"]
        if self.use_pruned_data:
            base_data_path = self.config["data"]["pruned_feature_selection_path"]
        else:
            base_data_path = self.config["data"]["unpruned_feature_selection_path"]
        pca_model_path = self.config["models"]["pca"]["model_path"]
        tsne_model_path = self.config["models"]["tsne"]["model_path"]
        sae_model_path = self.config["models"]["sae"]["model_path"]
        rf_results_path = self.config["models"]["random_forest"]["results_path"]

        tuned_params = self.tuned_parameters()
        dr_matrices, labels = self.load_data.get_dr_matrices(labels_path=labels_path, base_data_path=base_data_path,
                                                             pca_model_path=pca_model_path,
                                                             tsne_model_path=tsne_model_path,
                                                             sae_model_path=sae_model_path, num_rows=num_rows)
        dr_results_array = self.prepare_gridsearch(dr_matrices=dr_matrices, tuned_parameters=tuned_params,
                                                   labels=labels, rf_results_path=rf_results_path)
        dr_results_df = pd.DataFrame(dr_results_array)
        dr_results_df.to_msgpack(rf_results_path + "/" + "random_forest_gridsearch")
        self.log.info("GridSearch on Random Forest completed")
        self.log.info("Time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    random_forest_gs = RandomForestGridSearch(multi_class=True, use_pruned_data=True)
    random_forest_gs.main(num_rows=346679)
