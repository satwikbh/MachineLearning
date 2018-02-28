from time import time
from sklearn import metrics

from Utils.LoggerUtil import LoggerUtil


class MultiClassMetrics:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    @staticmethod
    def confusion_matrix(labels_pred, labels_actual):
        cnf_matrix = metrics.confusion_matrix(y_true=labels_actual, y_pred=labels_pred)
        return cnf_matrix

    def metrics(self, labels_pred, labels_actual):
        average_list = ['micro', 'macro', 'weighted', 'samples']
        precision_dict = dict()
        recall_dict = dict()
        fbeta_score_dict = dict()
        support_dict = dict()

        for average in average_list:
            precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_true=labels_actual,
                                                                                              y_pred=labels_pred,
                                                                                              average=average)
            self.log.info(
                "Metrics for average : {} are : \nPrecision : {}\tRecall : {}\tF1_Score : {}\tSupport : {}".format(
                    average, precision, recall, fbeta_score, support))
            precision_dict[average] = precision
            recall_dict[average] = recall
            fbeta_score_dict[average] = fbeta_score
            support_dict[average] = support

        cnf_matrix = self.confusion_matrix(labels_actual=labels_actual, labels_pred=labels_pred)
        return precision_dict, recall_dict, fbeta_score_dict, support_dict, cnf_matrix

    def main(self, labels_pred, labels_actual):
        start_time = time()
        precision_dict, recall_dict, fbeta_score_dict, support_dict, cnf_matrix = self.metrics(
            labels_pred=labels_pred,
            labels_actual=labels_actual)
        self.log.info("Total time taken : {}".format(time() - start_time))
