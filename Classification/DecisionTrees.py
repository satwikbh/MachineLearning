from time import time

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from HelperFunctions.HelperFunction import HelperFunction
from HelperFunctions.LoadDRMatrices import LoadDRMatrices
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class DecisionTrees:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_dr_mat = LoadDRMatrices()
        self.helper = HelperFunction
        self.classifier_name = "decision_trees"

    def compute_metrics(self, y_test, y_pred, dr_name, dt_results_path):
        """

        :param y_test:
        :param y_pred:
        :param dr_name:
        :param dt_results_path:
        :return:
        """
        results = dict()
        cr_report = classification_report(y_pred=y_pred, y_true=y_test)
        key_name = dr_name + "_" + str(self.classifier_name) + "_" + "cr_report"
        results[key_name] = cr_report
        cnf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)

        self.log.info("Classification Report : \n{}".format(cr_report))
        plt = self.helper.plot_cnf_matrix(cnf_matrix=cnf_matrix)
        plt_path = dt_results_path + "/" + str(dr_name) + "_" + str(
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
        cv = 5
        self.log.info("Using Decision Trees classifier")
        clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
        clf.fit(train_data, train_labels)
        self.log.info("Performing cross validation with cv : {}".format(cv))
        scores = cross_val_score(clf, train_data, train_labels, cv=cv)
        self.log.info("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return clf

    def prepare_classifier(self, dr_matrices, labels, dt_results_path):
        """

        :param dr_matrices:
        :param labels:
        :param dt_results_path:
        :return:
        """
        dr_results_array = dict()
        for dr_name, dr_matrix in dr_matrices.items():
            if dr_name is "base_data":
                self.log.info("Using Decision Tree classifier on Base Data")
            elif dr_name is "pca":
                self.log.info("Using Decision Tree classifier on PCA")
            elif dr_name is "tsne_random":
                self.log.info("Using Decision Tree classifier on TSNE with random init")
            elif dr_name is "tsne_pca":
                self.log.info("Using Decision Tree classifier on TSNE with pca init")
            elif dr_name is "sae":
                self.log.info("Using Decision Tree classifier on on SAE")
            else:
                self.log.error("Dimensionality Reduction technique employed is not supported!!!")
            x_train, x_test, y_train, y_test = self.helper.validation_split(dr_matrix, labels, test_size=0.25)
            del dr_matrix, labels
            clf = self.classification(train_data=x_train,
                                      train_labels=y_train)
            y_pred = self.prediction(clf=clf, test_data=x_test)
            results = self.compute_metrics(y_test=y_test, y_pred=y_pred, dr_name=dr_name,
                                           dt_results_path=dt_results_path)

            dr_results_array[dr_name] = results
        return dr_results_array

    def main(self, num_rows):
        start_time = time()
        labels_path = self.config["data"]["labels_path"]
        base_data_path = self.config["data"]["feature_selection_path"]
        pca_model_path = self.config["models"]["pca"]["model_path"]
        tsne_model_path = self.config["models"]["tsne"]["model_path"]
        sae_model_path = self.config["models"]["sae"]["model_path"]
        dt_results_path = self.config["models"]["decision_trees"]["results_path"]

        dr_matrices, labels = self.load_dr_mat.get_dr_matrices(labels_path=labels_path, base_data_path=base_data_path,
                                                               pca_model_path=pca_model_path,
                                                               tsne_model_path=tsne_model_path,
                                                               sae_model_path=sae_model_path, num_rows=num_rows)
        self.prepare_classifier(dr_matrices=dr_matrices, labels=labels, dt_results_path=dt_results_path)
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    dt = DecisionTrees()
    dt.main(num_rows=50000)
