#!/usr/bin/env python
import sys


def tp_fp_fn(correct_set, guess_set):
    """
    INPUT: dictionary with the elements in the cluster from the ground truth
    (CORRECT_SET) and dictionary with the elements from the estimated cluster
    (ESTIMATED_SET).

    OUTPUT: number of True Positives (elements in both clusters), False
    Positives (elements only in the ESTIMATED_SET), False Negatives (elements
    only in the CORRECT_SET).
    """
    tp = 0
    fp = 0
    fn = 0
    for elem in guess_set:
        # True Positives (elements in both clusters)
        if elem in correct_set:
            tp += 1
        else:
            # False Positives (elements only in the "estimated cluster")
            fp += 1
    for elem in correct_set:
        if elem not in guess_set:
            # False Negatives (elements only in the "correct cluster")
            fn += 1
    return tp, fp, fn


def eval_precision_recall_f_measure(groundtruth_dict, estimated_dict):
    """
    INPUT: dictionary with the mapping "element:cluster_id" for both the ground
    truth and the ESTIMATED_DICT clustering.

    OUTPUT: average values of Precision, Recall and F-Measure.
    """
    # eval: precision, recall, f-measure
    tmp_precision = 0
    tmp_recall = 0

    # build reverse dictionary of ESTIMATED_DICT
    rev_est_dict = {}
    for k, v in estimated_dict.items():
        if v not in rev_est_dict:
            rev_est_dict[v] = {k}
        else:
            rev_est_dict[v].add(k)

    # build reverse dictionary of GROUNDTRUTH_DICT
    gt_rev_dict = {}
    for k, v in groundtruth_dict.items():
        if v not in gt_rev_dict:
            gt_rev_dict[v] = {k}
        else:
            gt_rev_dict[v].add(k)

    counter, length_11 = 0, len(estimated_dict)

    sys.stderr.write('Calculating precision and recall\n')

    # For each element
    for element in estimated_dict:

        # Print progress
        if counter % 1000 == 0:
            sys.stderr.write('\r%d out of %d' % (counter, length_11))
            sys.stderr.flush()
        counter += 1

        # Get elements in the same cluster (for "ESTIMATED_DICT cluster")
        guess_cluster_id = estimated_dict[element]

        # Get the list of elements in the same cluster ("correct cluster")
        correct_cluster_id = groundtruth_dict[element]

        # Calculate TP, FP, FN
        tp, fp, fn = tp_fp_fn(gt_rev_dict[correct_cluster_id],
                              rev_est_dict[guess_cluster_id])

        # tmp_precision
        p = 1.0 * tp / (tp + fp)
        tmp_precision += p
        # tmp_recall
        r = 1.0 * tp / (tp + fn)
        tmp_recall += r
    sys.stderr.write('\r%d out of %d' % (counter, length_11))
    sys.stderr.write('\n')
    precision = 100.0 * tmp_precision / len(estimated_dict)
    recall = 100.0 * tmp_recall / len(estimated_dict)
    f_measure = (2 * precision * recall) / (precision + recall)
    return precision, recall, f_measure


if __name__ == "__main__":

    # The ground truth.
    # Dictionary with mapping: "element : cluster_id".
    diz_grth = {
        "a": 1,
        "b": 1,
        "c": 2,
        "d": 3
    }

    # An example of an "estimated cluster".
    # Dictionary with mapping: "element : cluster_id".
    diz_estim = {
        "a": 66,
        "b": 'malware',
        "c": 'goodware',
        "d": 'trojan'
    }

    # An example of an "estimated cluster": same partitioning as for the ground
    # truth, but just different cluster labels. Precision == Recall ==
    # F-Measure == 100%.
    # Dictionary with mapping: "element : cluster_id".
    diz_estim_grth = {
        "a": 2,
        "b": 2,
        "c": 66,
        "d": 9
    }

    # a sample where estimated != ground truth
    print("Ground truth")
    print("%8s --> %10s" % ("Element", "Cluster_ID"))

    for k, v in diz_grth.items():
        print("%8s --> %10s" % (k, v))
    print("Estimated clustering")
    print("%8s --> %10s" % ("Element", "Cluster_ID"))

    for k, v in diz_estim.items():
        print("%8s --> %10s" % (k, v))
    # precision, recall, f-measure
    p, r, f = eval_precision_recall_f_measure(diz_grth, diz_estim)
    print("Precison: %s%%" % p)
    print("Recall: %s%%" % r)
    print("F-Measure: %s%%" % f)
