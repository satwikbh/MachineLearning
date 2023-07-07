from time import time

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from Utils.LoggerUtil import LoggerUtil


class MultiClassMetrics:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    @staticmethod
    def confusion_matrix(labels_pred, labels_actual):
        cnf_matrix = confusion_matrix(y_true=labels_actual, y_pred=labels_pred)
        return cnf_matrix

    def get_top_k(self, clf, x_test, y_test, k):
        test_start_index = 0
        test_iter = 0
        chunk_size = 1000
        top_k_acc_list = list()

        while test_start_index < x_test.shape[0]:
            if test_iter % chunk_size == 0:
                self.log.info("Iteration : #{}".format(test_iter))
            if test_start_index + chunk_size < x_test.shape[0]:
                p_matrix = x_test[test_start_index: test_start_index + chunk_size]
                p_labels = y_test[test_start_index: test_start_index + chunk_size]
            else:
                p_matrix = x_test[test_start_index:]
                p_labels = y_test[test_start_index:]
            for index, pred_prob in enumerate(clf.predict_proba(p_matrix)):
                x, y = p_labels[index], pred_prob.argsort()[-k:].tolist()
                if x in y:
                    top_k_acc = 1
                else:
                    top_k_acc = 0
                top_k_acc_list.append(top_k_acc)
            test_start_index += chunk_size
            test_iter += 1

        return top_k_acc_list

    def metrics(self, labels_pred, labels_actual):
        average_list = ['micro', 'macro', 'weighted', 'samples']
        precision_dict = dict()
        recall_dict = dict()
        fbeta_score_dict = dict()
        support_dict = dict()

        for average in average_list:
            precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true=labels_actual,
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
        self.metrics(labels_pred=labels_pred, labels_actual=labels_actual)
        self.log.info("Total time taken : {}".format(time() - start_time))
