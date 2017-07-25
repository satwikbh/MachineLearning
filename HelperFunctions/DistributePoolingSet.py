import hickle
import json

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from HelperFunction import HelperFunction


class DistributePoolingSet:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.config = ConfigUtil().get_config_instance()
        self.feature_vector_ext = "feature_vector_part-"

    def save_distributed_feature_vector(self, sub_matrix, sub_matrix_index):
        try:
            self.log.info("Storing the {} part of the object".format(sub_matrix_index))
            self.helper.create_dir_if_absent(self.config['data']['feature_vector_path'])
            file_name = self.config['data']['feature_vector_path'] + "/" + self.feature_vector_ext + str(sub_matrix_index) + ".hkl"
            hickle.dump(sub_matrix.tocsr(), file_name, mode='w', compression='gzip')
            return file_name
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def load_distributed_feature_vector(self, dist_fnames):
        try:
            path = self.config['data']['feature_vector_path']
            completed = self.helper.get_files_starts_with_extension(extension=self.feature_vector_ext, path=path)
            completed = [x.split("/")[-1].split("feature_vector_")[1] for x in completed]
            dist_fnames = [x.split("/")[-1].split("feature_pool_")[1] for x in dist_fnames]
            remaining = set(dist_fnames).difference(completed)
            self.log.info("Completed : {}\tRemaining : {}".format(len(completed), len(remaining)))
            remaining = [self.path + "/" + self.feature_pool_dir_name + "/" + self.feature_pool_ext.split("part")[0] + x
                         for x in
                         remaining]
            return list(remaining), len(completed)
        except Exception as e:
            self.log.error("Error : {}".format(e))
