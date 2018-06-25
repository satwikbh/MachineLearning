from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from PrepareData.LoadData import LoadData
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class ClassificationPipeline:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()
        self.scores = ['precision', 'recall']

    @staticmethod
    def set_estimators(num_features):
        estimators = list()
        estimators.append(('feature_selection', SelectKBest(chi2, k=num_features)))
        estimators.append(('pca', PCA()))
        estimators.append(('ovr', OneVsRestClassifier(SVC())))
        return estimators

    @staticmethod
    def prepare_pipeline(estimators):
        pipe = Pipeline(estimators)
        return pipe

    @staticmethod
    def get_gridsearch_params(num_features):
        param_grid = dict()
        param_grid['pca__n_components'] = range(num_features / 4, num_features / 2, 2)
        param_grid['ovr__estimator__c'] = [0.01, 0.1, 1.0, 10, 100]
        param_grid['ovr__estimator__kernel'] = ['rbf']
        param_grid['ovr__estimator__gamma'] = [1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3, 1e-4]
        return param_grid

    def perform_gridsearch(self, pipe, param_grid, x_train, x_test, y_train, y_test):
        for score in self.scores:
            grid = GridSearchCV(pipe, cv=3, n_jobs=-1, param_grid=param_grid, scoring='%s_macro' % score)
            grid.fit(X=x_train.toarray(), y=y_train)
            self.log.info("Best parameters set found on development set : \n{}".format(grid.best_params_))
            means = grid.cv_results_['mean_test_score']
            stds = grid.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, grid.cv_results_['params']):
                self.log.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            self.log.info("Detailed classification report:")
            self.log.info("The model is trained on the full development set.")
            self.log.info("The scores are computed on the full evaluation set.")
            y_true, y_pred = y_test, grid.predict(x_test.toarray())
            self.log.info(classification_report(y_true, y_pred))

    @staticmethod
    def test_split(input_matrix, labels):
        x_train, x_test, y_train, y_test = train_test_split(input_matrix, labels, test_size=0.25, random_state=13)
        return x_train, x_test, y_train, y_test

    def get_pruning_threshold(self, threshold, input_matrix):
        try:
            sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
            pruned_input = sel.fit_transform(input_matrix)
            return pruned_input.shape
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def main(self, num_rows):
        fv_path = self.config['data']['feature_vector_path']
        labels_path = self.config['data']['labels_path']
        pruning_threshold = self.config['data']['pruning_threshold']
        input_matrix, input_matrix_indices, labels = self.load_data.get_data_with_labels(num_rows=num_rows,
                                                                                         data_path=fv_path,
                                                                                         labels_path=labels_path)
        num_variants, num_features = self.get_pruning_threshold(threshold=pruning_threshold, input_matrix=input_matrix)
        self.log.info("Features pruned from {} to {}".format(input_matrix.shape[1], num_features))
        estimators = self.set_estimators(num_features=num_features)
        x_train, x_test, y_train, y_test = self.test_split(input_matrix=input_matrix, labels=labels)
        pipe = self.prepare_pipeline(estimators=estimators)
        param_grid = self.get_gridsearch_params(num_features=num_features)
        self.perform_gridsearch(pipe=pipe,
                                param_grid=param_grid,
                                x_train=x_train,
                                x_test=x_test,
                                y_train=y_train,
                                y_test=y_test)


if __name__ == '__main__':
    cp = ClassificationPipeline()
    cp.main(num_rows=28000)
