from time import time

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from HelperFunctions.HelperFunction import HelperFunction
from HelperFunctions.LoadDRMatrices import LoadDRMatrices
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class Adaboost:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_dr_mat = LoadDRMatrices()
        self.helper = HelperFunction()
        self.classifier_name = "adaboost"

    def perform_classification(self, input_matrix, dr_name, labels, adaboost_results_path):
        """

        :param input_matrix:
        :param dr_name:
        :param labels:
        :param adaboost_results_path:
        :return:
        """
        results = dict()
        cv = 5
        x_train, x_test, y_train, y_test = self.helper.validation_split(input_matrix, labels, test_size=0.25)
        del input_matrix, labels
        start_time = time()
        self.log.info("Using Adaboost classifier")
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        self.log.info("Performing cross validation with cv : {}".format(cv))
        scores = cross_val_score(clf, x_train, y_train, cv=cv)
        self.log.info("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        clf.fit(X=x_train, y=y_train)
        y_pred = clf.predict(X=x_test)

        cr_report = classification_report(y_pred=y_pred, y_true=y_test)
        key_name = dr_name + "_" + str(self.classifier_name) + "_" + "cr_report"
        results[key_name] = cr_report
        cnf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)

        self.log.info("Classification Report : \n{}".format(cr_report))
        plt = self.helper.plot_cnf_matrix(cnf_matrix=cnf_matrix)
        plt_path = adaboost_results_path + "/" + str(dr_name) + "_" + str(
            self.classifier_name) + "_" + "cnf_matrix" + ".png"
        self.log.info("Saving Confusion Matrix at path : {}".format(plt_path))
        plt.savefig(plt_path)
        self.log.info("Time taken : {}".format(time() - start_time))
        return results

    def prepare_classifier(self, dr_matrices, labels, adaboost_results_path):
        """

        :param dr_matrices:
        :param labels:
        :param adaboost_results_path:
        :return:
        """
        dr_results_array = dict()
        for dr_name, dr_matrix in dr_matrices.items():
            if dr_name is "base_data":
                self.log.info("Using Adaboost classifier on Base Data")
            elif dr_name is "pca":
                self.log.info("Using Adaboost classifier on PCA")
            elif dr_name is "tsne_random":
                self.log.info("Using Adaboost classifier on TSNE with random init")
            elif dr_name is "tsne_pca":
                self.log.info("Using Adaboost classifier on TSNE with pca init")
            elif dr_name is "sae":
                self.log.info("Using Adaboost classifier on on SAE")
            else:
                self.log.error("Dimensionality Reduction technique employed is not supported!!!")
            dr_results_array[dr_name] = self.perform_classification(input_matrix=dr_matrix,
                                                                    dr_name=dr_name,
                                                                    labels=labels,
                                                                    adaboost_results_path=adaboost_results_path)
        return dr_results_array

    def main(self, num_rows):
        """

        :param num_rows:
        :return:
        """
        start_time = time()
        labels_path = self.config["data"]["labels_path"]
        base_data_path = self.config["data"]["feature_selection_path"]
        pca_model_path = self.config["models"]["pca"]["model_path"]
        tsne_model_path = self.config["models"]["tsne"]["model_path"]
        sae_model_path = self.config["models"]["sae"]["model_path"]
        adaboost_results_path = self.config["models"]["adaboost"]["results_path"]

        dr_matrices, labels = self.load_dr_mat.get_dr_matrices(labels_path=labels_path, base_data_path=base_data_path,
                                                               pca_model_path=pca_model_path,
                                                               tsne_model_path=tsne_model_path,
                                                               sae_model_path=sae_model_path, num_rows=num_rows)
        self.prepare_classifier(dr_matrices=dr_matrices, labels=labels, adaboost_results_path=adaboost_results_path)
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    adaboost_clf = Adaboost()
    adaboost_clf.main(num_rows=50000)
