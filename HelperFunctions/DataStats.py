from time import time

import numpy as np
from scipy.sparse import load_npz, save_npz, vstack
from sklearn.feature_selection import VarianceThreshold

from HelperFunctions.HelperFunction import HelperFunction
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class DataStats:
    def __init__(self, use_trie_pruning):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil().get_config_instance()
        self.helper = HelperFunction()
        self.use_trie_pruning = use_trie_pruning

    @staticmethod
    def stats(fv):
        fv_csc = fv.tocsc()
        index_pointer = fv_csc.indptr
        new_index_pointer = np.diff(index_pointer)
        return new_index_pointer

    def sum_up(self, partial_index_pointer, col_wise_dist):
        self.log.info(F"partial_index_pointer : {len(partial_index_pointer)}\tcol_wise_dist : {len(col_wise_dist)}")
        if len(col_wise_dist) == 0:
            return partial_index_pointer
        else:
            assert len(partial_index_pointer) == len(col_wise_dist)
            for x in range(len(partial_index_pointer)):
                col_wise_dist[x] = col_wise_dist[x] + partial_index_pointer[x]
            return col_wise_dist

    def get_stats(self, list_of_files):
        col_wise_dist = list()
        num_rows = 0
        for index, each_file in enumerate(list_of_files):
            self.log.info(F"Iteration : {index}")
            fv = load_npz(each_file)
            num_rows += fv.shape[0]
            partial_index_pointer = self.stats(fv)
            col_wise_dist = self.sum_up(partial_index_pointer, col_wise_dist)
            del fv
        return col_wise_dist, num_rows

    @staticmethod
    def remove_duplicates(old_mat):
        new_mat = old_mat.tocsr()
        data = new_mat.data
        new_data = list()
        for c_data in data:
            if c_data == 1 or c_data == 0:
                new_data.append(c_data)
            else:
                new_data.append(1)
        new_mat.data = np.asarray(new_data, dtype=np.float32)
        return new_mat

    def delete_columns(self, old_mat, cols_to_delete):
        all_cols = np.arange(old_mat.shape[1])
        cols_to_keep = np.where(np.logical_not(np.in1d(all_cols, cols_to_delete)))[0]
        old_mat = self.remove_duplicates(old_mat)
        new_mat = old_mat[:, cols_to_keep]
        new_mat = new_mat.astype(np.float32)
        return new_mat

    @staticmethod
    def estimate_cols_to_remove(col_wise_dist, threshold):
        cols_to_delete = list()
        for index, value in enumerate(col_wise_dist):
            if value < threshold:
                cols_to_delete.append(index)
        return cols_to_delete

    def store_pruned_matrix(self, feature_vector, col_wise_dist, pruned_matrix_path, num_rows):
        threshold = num_rows * self.config['data']['pruning_threshold']
        cols_to_delete = self.estimate_cols_to_remove(col_wise_dist, threshold)
        for index, each_file in enumerate(feature_vector):
            fv = load_npz(each_file)
            new_mat = self.delete_columns(fv, cols_to_delete)
            file_name = pruned_matrix_path + "/" + "pruned_mat_part_" + str(index)
            save_npz(file_name, new_mat, compressed=True)

    def perform_feature_selection(self, feature_vector_file_list, variance_threshold, feature_selection_path,
                                  chunk_size):
        self.log.info("Performing Feature Selection on the feature vector")
        threshold_meta = variance_threshold * (1 - variance_threshold)
        sel = VarianceThreshold(threshold=threshold_meta)
        fv_list = list()
        for each_file in feature_vector_file_list:
            fv_list.append(load_npz(each_file))
        feature_vector = vstack(fv_list)
        feature_selection = sel.fit_transform(feature_vector)
        count = 0
        index = 0
        while count < feature_selection.shape[0]:
            if count + chunk_size < feature_selection.shape[0]:
                p_matrix = feature_selection[count: count + chunk_size]
            else:
                p_matrix = feature_selection[count:]
            file_name = feature_selection_path + "/" + "feature_selection_part_" + str(index)
            self.log.info(F"Iter : #{index}")
            save_npz(file_name, p_matrix, compressed=True)
            count += chunk_size
            index += 1

    def main(self):
        start_time = time()
        self.log.info("Generating column wise count of non zero elements")

        unpruned_feature_vector_path = self.config["data"]["unpruned_feature_vector_path"]
        pruned_feature_vector_path = self.config["data"]["pruned_feature_vector_path"]

        pruned_variance_matrix_path = self.config["data"]["pruned_variance_matrix_path"]
        unpruned_variance_matrix_path = self.config["data"]["unpruned_variance_matrix_path"]

        pruned_feature_selection_path = self.config["data"]["pruned_feature_selection_path"]
        unpruned_feature_selection_path = self.config["data"]["unpruned_feature_selection_path"]

        col_dist_path = self.config['data']['col_dist_path']
        chunk_size = self.config['data']['config_param_chunk_size']
        variance_threshold = self.config['data']['variance_threshold']

        if self.use_trie_pruning:
            self.helper.create_dir_if_absent(pruned_feature_vector_path)
            self.helper.create_dir_if_absent(pruned_variance_matrix_path)
            self.helper.create_dir_if_absent(pruned_feature_selection_path)
            pruned_feature_vector_file_list = self.helper.get_files_ends_with_extension(path=pruned_feature_vector_path,
                                                                                        extension=".npz")
            self.log.info(F"Total number of files : {len(pruned_feature_vector_file_list)}")
            col_wise_dist, num_rows = self.get_stats(pruned_feature_vector_file_list)
            np.savez(col_dist_path + "/" + "col_wise_dist.dump", np.asarray(col_wise_dist))
            self.store_pruned_matrix(feature_vector=pruned_feature_vector_file_list, col_wise_dist=col_wise_dist,
                                     pruned_matrix_path=pruned_variance_matrix_path, num_rows=num_rows)
            self.perform_feature_selection(feature_vector_file_list=pruned_feature_vector_file_list,
                                           variance_threshold=variance_threshold, chunk_size=chunk_size,
                                           feature_selection_path=pruned_feature_selection_path)
        else:
            self.helper.create_dir_if_absent(unpruned_feature_vector_path)
            self.helper.create_dir_if_absent(unpruned_variance_matrix_path)
            self.helper.create_dir_if_absent(unpruned_feature_selection_path)
            unpruned_feature_vector_file_list = self.helper.get_files_ends_with_extension(
                path=unpruned_feature_vector_path,
                extension=".npz")
            self.log.info(F"Total number of files : {len(unpruned_feature_vector_file_list)}")
            col_wise_dist, num_rows = self.get_stats(unpruned_feature_vector_file_list)
            np.savez(col_dist_path + "/" + "unpruned_col_wise_dist.dump", np.asarray(col_wise_dist))
            self.store_pruned_matrix(feature_vector=unpruned_feature_vector_file_list, col_wise_dist=col_wise_dist,
                                     pruned_matrix_path=unpruned_variance_matrix_path, num_rows=num_rows)
            self.perform_feature_selection(feature_vector_file_list=unpruned_feature_vector_file_list,
                                           variance_threshold=variance_threshold, chunk_size=chunk_size,
                                           feature_selection_path=unpruned_feature_selection_path)

        self.log.info(F"Total time for execution : {time() - start_time}")
