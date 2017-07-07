import hickle
import numpy as np
import json

from Utils.LoggerUtil import LoggerUtil
from HelperFunction import HelperFunction


class DistributePoolingSet:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.path = "/home/satwik/Documents/Research/MachineLearning/HelperFunctions"
        self.feature_pool_dir_name = "feature_pool"
        self.feature_pool_ext = "feature_pool_part-"
        self.feature_vector_dir_name = "feature_vector"
        self.feature_vector_ext = "feature_vector_part-"

    def save_distributed_pool(self, mini_batch, part):
        try:
            self.log.info("Storing the {} part of the object".format(part))
            self.helper.create_dir_if_absent(self.feature_pool_dir_name)
            file_name = self.path + "/" + self.feature_pool_dir_name + "/" + self.feature_pool_ext + str(part) + ".hkl"
            json_dump_value = json.dumps(mini_batch)
            hickle.dump(json_dump_value, file_name, mode='w', compression='gzip')
            return file_name
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def save_distributed_feature_vector(self, mini_batch, part):
        try:
            self.log.info("Storing the {} part of the object".format(part))
            self.helper.create_dir_if_absent(self.feature_vector_dir_name)
            file_name = self.path + "/" + self.feature_vector_dir_name + "/" + self.feature_vector_ext + str(
                part) + ".hkl"
            hickle.dump(mini_batch.tocsr(), file_name, mode='w', compression='gzip')
            return file_name
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def load_distributed_feature_vector(self, dist_fnames):
        try:
            path = self.path + "/" + self.feature_vector_dir_name + "/"
            completed = self.helper.get_files_with_extension(extension=self.feature_vector_ext, path=path)
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

    def load_distributed_pool(self):
        try:
            path = self.path + "/" + self.feature_pool_dir_name + "/"
            completed = self.helper.get_files_with_extension(extension=self.feature_pool_ext, path=path)
            self.log.info("Total number of feature pool chunks generated : {}".format(len(completed)))
            return completed
        except Exception as e:
            self.log.error("Error : {}".format(e))
