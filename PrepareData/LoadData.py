from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from HelperFunctions.HelperFunction import HelperFunction


class LoadData:
    """
    This class is used to load the data once it is prepared.
    Incase the data is not prepared run "PrepareDataset"
    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.config = ConfigUtil.get_config_instance()

    def main(self, num_rows):
        num_files = num_rows / 1000
        input_matrix_indices = range(num_rows)
        pruned_fv_path = self.config['data']['pruned_feature_vector_path']
        list_of_files = self.helper.get_files_ends_with_extension(".hkl", pruned_fv_path)
        matrix = self.helper.open_files(list_of_files[:num_files])
        input_matrix = self.helper.stack_matrix(matrix)
        nearest_repr = self.helper.nearest_power_of_two(input_matrix.shape[1])
        self.log.info("Input matrix dimension : {}\tNearest power of 2 : {}".format(input_matrix.shape, nearest_repr))
        return input_matrix, input_matrix_indices
