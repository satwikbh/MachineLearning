import numpy as np

from libact.query_strategies.multiclass import ActiveLearningWithCostEmbedding, HierarchicalSampling, \
    expected_error_reduction, UncertaintySampling
from libact.query_strategies import ActiveLearningByLearning
from libact.models import SVM
from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler
from sklearn.model_selection import train_test_split

from PrepareData.LoadData import LoadData
from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil


class LibactAL:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()

    def prepare_datagen(self, input_matrix, labels, n_labeled):
        data_train, data_test, labels_train, labels_test = train_test_split(input_matrix, labels, test_size=0.25,
                                                                            random_state=13)

        trn_ds = Dataset(data_train,
                         np.concatenate([labels_train[:n_labeled], [None] * (len(labels_train) - n_labeled)]))
        tst_ds = Dataset(data_test, np.concatenate([labels_test[:n_labeled], [None] * (len(labels_test) - n_labeled)]))
        fully_labeled_trn_ds = Dataset(data_train, labels_train)
        return trn_ds, tst_ds, labels_train, fully_labeled_trn_ds

    def run(self, trn_ds, tst_ds, lbr, model, qs, quota):
        E_in, E_out = [], []

        for _ in range(quota):
            # Standard usage of libact objects
            ask_id = qs.make_query()
            X, _ = zip(*trn_ds.data)
            lb = lbr.label(X[ask_id])
            trn_ds.update(ask_id, lb)

            model.train(trn_ds)
            E_in = np.append(E_in, 1 - model.score(trn_ds))
            E_out = np.append(E_out, 1 - model.score(tst_ds))

        return E_in, E_out

    def main(self, num_rows):
        pruned_data_path = self.config['data']['pruned_fv_path']
        labels_path = self.config['data']['labels_path']
        input_matrix, input_matrix_indices, labels = self.load_data.get_data_with_labels(num_rows=num_rows,
                                                                                         data_path=pruned_data_path,
                                                                                         labels_path=labels_path)
        n_labeled = int(len(labels) * 0.33)
        trn_ds, tst_ds, labels_train, fully_labeled_trn_ds = self.prepare_datagen(input_matrix=input_matrix,
                                                                                  labels=labels,
                                                                                  n_labeled=n_labeled)
        lbr = IdealLabeler(fully_labeled_trn_ds)
        quota = len(labels_train) - n_labeled
        model = SVM()

        sub_qs = UncertaintySampling(trn_ds, method='sm', model=SVM(decision_function_shape='ovr'))

        qs = HierarchicalSampling(
            trn_ds,  # Dataset object
            trn_ds.get_num_of_labels(),
            active_selecting=True,
            subsample_qs=sub_qs
        )

        E_in, E_out = self.run(trn_ds=trn_ds, tst_ds=tst_ds, lbr=lbr, model=model, qs=qs, quota=quota)
        self.log.info("E_in : {}\nE_out : {}".format(E_in, E_out))


if __name__ == '__main__':
    active_learning = LibactAL()
    active_learning.main(num_rows=25000)
