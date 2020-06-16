import pickle as pi

from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler

from HelperFunctions.HelperFunction import HelperFunction
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class LoadData:
    """
    This class is used to load the data once it is prepared.
    In case the data is not prepared run "PrepareDataset"
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
        chunk_size = 1000
        if num_rows % chunk_size > 0:
            num_files = (num_rows / chunk_size) + 1
        else:
            num_files = num_rows / chunk_size
        list_of_files = self.helper.get_files_ends_with_extension(".npz", data_path)
        matrix = self.helper.open_np_files(list_of_files[:num_files])
        input_matrix = self.helper.stack_matrix(matrix)
        input_matrix_indices = range(input_matrix.shape[0])
        input_matrix = self.scale(input_matrix)
        nearest_repr = self.helper.nearest_power_of_two(input_matrix.shape[1])
        self.log.info(F"Input matrix dimension : {input_matrix.shape}\tNearest power of 2 : {nearest_repr}")
        labels = pi.load(open(labels_path + "/" + "labels.pkl"))
        return input_matrix, input_matrix_indices, labels

    def load_freq_top_k_data(self, num_rows, labels_path):
        files_fv_path = self.config["freq_individual_feature_vector_path"]["files_feature"]
        reg_keys_fv_path = self.config["freq_individual_feature_vector_path"]["reg_keys_feature"]
        mutexes_fv_path = self.config["freq_individual_feature_vector_path"]["mutexes_feature"]
        exec_cmds_fv_path = self.config["freq_individual_feature_vector_path"]["exec_cmds_feature"]
        network_fv_path = self.config["freq_individual_feature_vector_path"]["network_feature"]
        static_fv_path = self.config["freq_individual_feature_vector_path"]["static_feature"]

        files_matrix = self.helper.load_sparse_matrix(file_path=files_fv_path, num_rows=num_rows,
                                                      identifier="feature_vector_part_")
        reg_keys_matrix = self.helper.load_sparse_matrix(file_path=reg_keys_fv_path, num_rows=num_rows,
                                                         identifier="feature_vector_part_")
        mutexes_matrix = self.helper.load_sparse_matrix(file_path=mutexes_fv_path, num_rows=num_rows,
                                                        identifier="feature_vector_part_")
        exec_cmds_matrix = self.helper.load_sparse_matrix(file_path=exec_cmds_fv_path, num_rows=num_rows,
                                                          identifier="feature_vector_part_")
        network_matrix = self.helper.load_sparse_matrix(file_path=network_fv_path, num_rows=num_rows,
                                                        identifier="feature_vector_part_")
        static_matrix = self.helper.load_sparse_matrix(file_path=static_fv_path, num_rows=num_rows,
                                                       identifier="feature_vector_part_")
        matrix = hstack(
            [files_matrix, reg_keys_matrix, mutexes_matrix, exec_cmds_matrix, network_matrix, static_matrix])
        self.log.info(F"Feature vector shape (n_samples, n_features) : {matrix.shape}")
        labels = pi.load(open(labels_path + "/" + "labels.pkl"))
        self.log.info(F"Total number of labels are : {len(labels)}")
        return matrix, labels

    def main(self, num_rows):
        chunk_size = 1000
        if num_rows % chunk_size > 0:
            num_files = (num_rows / chunk_size) + 1
        else:
            num_files = num_rows / chunk_size
        feature_selection_path = self.config['data']['feature_selection_path']
        list_of_files = self.helper.get_files_ends_with_extension(".npz", feature_selection_path)
        matrix = self.helper.open_np_files(list_of_files[:num_files])
        input_matrix = self.helper.stack_matrix(matrix)
        input_matrix_indices = range(input_matrix.shape[0])
        input_matrix = self.scale(input_matrix)
        nearest_repr = self.helper.nearest_power_of_two(input_matrix.shape[1])
        self.log.info(F"Input matrix dimension : {input_matrix.shape}\tNearest power of 2 : {nearest_repr}")
        return input_matrix, input_matrix_indices
