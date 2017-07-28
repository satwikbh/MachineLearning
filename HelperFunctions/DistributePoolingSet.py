import hickle as hkl
import pickle as pi

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from HelperFunction import HelperFunction


class DistributePoolingSet:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.config = ConfigUtil().get_config_instance()
        self.feature_vector_ext = "feature_vector_main"
        self.feature_pool_ext = "feature_pool_main"

    def save_feature_vector(self, feature_vector):
        try:

            feature_vector_path = self.config['data']['feature_vector_path']
            file_name = feature_vector_path + "/" + self.feature_vector_ext + ".hkl"
            hkl.dump(feature_vector.tocsr(), file_name, mode='w', compression='gzip')
            return file_name
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def save_feature_pool(self, feature_pool_path, values, index):
        try:
            file_object = open(feature_pool_path + "/" + "feature_pool_part_" + str(index) + ".hkl", "w")
            pi.dump(values, file_object)
            file_object.close()
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def save_distributed_feature_vector(self, mini_batch_matrix, feature_vector_path, index):
        try:
            file_name = feature_vector_path + "/" + "feature_vector_part_" + str(index) + ".hkl"
            file_object = open(file_name, "w")
            hkl.dump(mini_batch_matrix.tocsr(), file_object, mode="w")
            file_object.close()
            return file_name
        except Exception as e:
            self.log.error("Error : {}".format(e))
