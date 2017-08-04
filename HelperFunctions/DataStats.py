from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from HelperFunctions.HelperFunction import HelperFunction
from time import time

import hickle as hkl
import numpy as np


class DataStats:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil().get_config_instance()
        self.helper = HelperFunction()

    @staticmethod
    def stats(fv):
        fv_csc = fv.tocsc()
        index_pointer = fv_csc.indptr
        new_index_pointer = np.diff(index_pointer)
        return new_index_pointer

    def sum_up(self, partial_index_pointer, col_wise_dist):
        self.log.info(
            "partial_index_pointer : {}\tcol_wise_dist : {}".format(len(partial_index_pointer), len(col_wise_dist)))
        if len(col_wise_dist) == 0:
            return partial_index_pointer
        else:
            assert len(partial_index_pointer) == len(col_wise_dist)
            for x in xrange(len(partial_index_pointer)):
                col_wise_dist[x] = col_wise_dist[x] + partial_index_pointer[x]
            return col_wise_dist

    def get_stats(self, list_of_files):
        col_wise_dist = list()
        num_rows = 0
        for index, each_file in enumerate(list_of_files):
            self.log.info("Iteration : {}".format(index))
            fv = hkl.load(each_file)
            num_rows += fv.shape[0]
            partial_index_pointer = self.stats(fv)
            col_wise_dist = self.sum_up(partial_index_pointer, col_wise_dist)
            del fv
        return col_wise_dist, num_rows

    @staticmethod
    def delete_columns(old_mat, cols_to_delete):
        all_cols = np.arange(old_mat.shape[1])
        cols_to_keep = np.where(np.logical_not(np.in1d(all_cols, cols_to_delete)))[0]
        new_mat = old_mat[:, cols_to_keep]
        return new_mat

    @staticmethod
    def estimate_cols_to_remove(col_wise_dist, threshold):
        cols_to_delete = list()
        # threshold = np.mean(col_wise_dist)
        for index, value in enumerate(col_wise_dist):
            if value < threshold:
                cols_to_delete.append(index)
        return cols_to_delete

    def store_pruned_matrix(self, feature_vector, col_wise_dist, pruned_matrix_path, num_rows):
        threshold = num_rows * self.config['data']['pruning_threshold']
        cols_to_delete = self.estimate_cols_to_remove(col_wise_dist, threshold)
        for index, each_file in enumerate(feature_vector):
            fv = hkl.load(each_file).tocsr()
            new_mat = self.delete_columns(fv, cols_to_delete)
            file_name = pruned_matrix_path + "/" + "pruned_mat_part_" + str(index) + ".hkl"
            hkl.dump(new_mat, open(file_name, "w"))

    def main(self):
        start_time = time()
        self.log.info("Generating column wise count of non zero elements")
        feature_vector_path = self.config['data']['feature_vector_path']
        pruned_matrix_path = self.config['data']['pruned_feature_vector_path']
        col_dist_path = self.config['data']['col_dist_path']

        self.helper.create_dir_if_absent(feature_vector_path)
        self.helper.create_dir_if_absent(pruned_matrix_path)

        feature_vector = self.helper.get_files_ends_with_extension(path=feature_vector_path, extension=".hkl")
        self.log.info("Total number of files : {}".format(len(feature_vector)))
        col_wise_dist, num_rows = self.get_stats(feature_vector)
        hkl.dump(np.asarray(col_wise_dist), open(col_dist_path + "/" + "col_wise_dist.dump", "w"))
        self.store_pruned_matrix(feature_vector, col_wise_dist, pruned_matrix_path, num_rows)
        self.log.info("Total time for execution : {}".format(time() - start_time))
