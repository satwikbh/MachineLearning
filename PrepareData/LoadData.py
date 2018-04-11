import pickle as pi

from sklearn.preprocessing import StandardScaler

from HelperFunctions.HelperFunction import HelperFunction
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class LoadData:
    """
    This class is used to load the data once it is prepared.
    Incase the data is not prepared run "PrepareDataset"
    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.config = ConfigUtil.get_config_instance()

    def scale(self, input_matrix):
        self.log.info("Performing Scaling on the input data")
        scaler = StandardScaler(with_mean=False).fit(input_matrix)
        return scaler.transform(input_matrix)

    def get_data_with_labels(self, num_rows, data_path, labels_path):
        if num_rows % 1000 > 0:
            num_files = (num_rows / 1000) + 1
        else:
            num_files = num_rows / 1000
        list_of_files = self.helper.get_files_ends_with_extension(".npz", data_path)
        matrix = self.helper.open_np_files(list_of_files[:num_files])
        input_matrix = self.helper.stack_matrix(matrix)
        input_matrix_indices = range(input_matrix.shape[0])
        input_matrix = self.scale(input_matrix)
        nearest_repr = self.helper.nearest_power_of_two(input_matrix.shape[1])
        self.log.info("Input matrix dimension : {}\tNearest power of 2 : {}".format(input_matrix.shape, nearest_repr))
        labels = pi.load(open(labels_path + "/" + "labels.pkl"))
        return input_matrix, input_matrix_indices, labels

    def main(self, num_rows):
        if num_rows % 1000 > 0:
            num_files = (num_rows / 1000) + 1
        else:
            num_files = num_rows / 1000
        pruned_fv_path = self.config['data']['pruned_feature_vector_path']
        feature_selection_path = self.config['data']['feature_selection_path']
        list_of_files = self.helper.get_files_ends_with_extension(".npz", feature_selection_path)
        matrix = self.helper.open_np_files(list_of_files[:num_files])
        input_matrix = self.helper.stack_matrix(matrix)
        input_matrix_indices = range(input_matrix.shape[0])
        input_matrix = self.scale(input_matrix)
        nearest_repr = self.helper.nearest_power_of_two(input_matrix.shape[1])
        self.log.info("Input matrix dimension : {}\tNearest power of 2 : {}".format(input_matrix.shape, nearest_repr))
        return input_matrix, input_matrix_indices
