import hickle as hkl
import pickle as pi

from scipy.sparse import save_npz

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from HelperFunctions.HelperFunction import HelperFunction


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
            file_object = open(feature_pool_path + "/" + "feature_pool_part_" + str(index) + ".dump", "w")
            pi.dump(values, file_object)
            file_object.close()
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def save_distributed_feature_vector(self, mini_batch_matrix, feature_vector_path, index):
        try:
            file_name = feature_vector_path + "/" + "feature_vector_part_" + str(index) + ".npz"
            file_object = open(file_name, "w")
            save_npz(file_object, mini_batch_matrix, compressed=True)
            file_object.close()
            return file_name
        except Exception as e:
            self.log.error("Error : {}".format(e))
