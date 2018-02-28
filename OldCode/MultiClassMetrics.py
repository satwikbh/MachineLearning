import numpy as np
from sklearn.metrics import confusion_matrix

from Utils.LoggerUtil import LoggerUtil


class MultiClassMetrics:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    @staticmethod
    def get_tp(labels_pred, labels_actual, target_label):
        """
        If the predicted and the actual are equal to the target label.
        :param labels_pred:
        :param labels_actual:
        :param target_label:
        :return:
        """
        return [x for x in xrange(len(labels_pred)) if
                (labels_pred[x] == labels_actual[x]) and labels_pred[x] == target_label]

    @staticmethod
    def get_fp(labels_pred, target_label):
        """
        All the instances which are classified as target label whether true or not.
        :return:
        """
        return [x for x in xrange(len(labels_pred)) if labels_pred[x] == target_label]

    @staticmethod
    def get_fn(labels_actual, target_label):
        """
        All the instances which are actually target_label.
        :param labels_actual:
        :param target_label:
        :return:
        """
        return [x for x in xrange(len(labels_actual)) if labels_actual[x] == target_label]

    @staticmethod
    def micro_averaging(tp_dict, fp_dict, fn_dict):
        tp_sum = np.sum(tp_dict.values())
        fp_sum = np.sum(fp_dict.values())
        fn_sum = np.sum(fn_dict.values())

        precision_micro = tp_sum * 1.0 / (tp_sum + fp_sum)
        recall_micro = tp_sum * 1.0 / (tp_sum + fn_sum)
        f1_score_micro = 2 * ((precision_micro * recall_micro) / (precision_micro + recall_micro))
        return precision_micro, recall_micro, f1_score_micro

    @staticmethod
    def macro_averaging(tp_dict, fp_dict, fn_dict):
        precision_list = list()
        recall_list = list()
        f1_score_list = list()

        for target_label in tp_dict.keys():
            tp = tp_dict[target_label]
            fp = fp_dict[target_label]
            fn = fn_dict[target_label]

            precision = tp * 1.0 / (tp + fp)
            recall = tp * 1.0 / (fn + tp)
            f1_score = 2 * ((precision * recall) / (precision + recall))

            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

        precision_macro = np.mean(precision_list)
        recall_macro = np.mean(recall_list)
        f1_score_macro = np.mean(f1_score_list)

        return precision_macro, recall_macro, f1_score_macro

    def metrics(self, labels_pred, labels_actual):
        labels_actual_set = np.unique(labels_actual)
        labels_pred_set = np.unique(labels_pred)

        tp_dict = dict()
        fp_dict = dict()
        fn_dict = dict()

        for target_label in labels_actual_set:
            if target_label in labels_pred_set:
                tp = self.get_tp(labels_pred=labels_pred, labels_actual=labels_actual, target_label=target_label)
                fp = self.get_fp(labels_pred=labels_pred, target_label=target_label)
                fn = self.get_fn(labels_actual=labels_actual, target_label=target_label)
            else:
                tp, fp, fn = 0.0, 0.0, 0.0
            tp_dict[target_label] = tp
            fp_dict[target_label] = fp
            fn_dict[target_label] = fn

        return tp_dict, fp_dict, fn_dict

    @staticmethod
    def get_confusion_matrix(labels_actual, labels_pred):
        cnf_matrix = confusion_matrix(y_true=labels_actual, y_pred=labels_pred)
        return cnf_matrix

    def main(self, labels_pred, labels_actual):
        tp_dict, fp_dict, fn_dict = self.metrics(labels_pred=labels_pred, labels_actual=labels_actual)
        precision_micro, recall_micro, f1_score_micro = self.micro_averaging(tp_dict, fp_dict, fn_dict)
        precision_macro, recall_macro, f1_score_macro = self.macro_averaging(tp_dict, fp_dict, fn_dict)
        confusion_matrix = self.get_confusion_matrix(labels_actual=labels_actual, labels_pred=labels_pred)



if __name__ == "__main__":
    mcm = MultiClassMetrics()
    mcm.main()
