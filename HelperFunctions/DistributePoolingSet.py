import hickle
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
            self.helper.create_dir_if_absent(self.config['data']['feature_vector_path'])
            file_name = self.config['data']['feature_vector_path'] + "/" + self.feature_vector_ext + ".hkl"
            hickle.dump(feature_vector.tocsr(), file_name, mode='w', compression='gzip')
            return file_name
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def save_feature_pool(self, feature_pool_path, values, iteration):
        file_name = open(feature_pool_path + "/" + "feature_pool_part_" + str(iteration) + ".hkl", "w")
        pi.dump(values, file_name)
        file_name.close()
