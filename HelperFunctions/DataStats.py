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
        new_index_pointer = list()

        for x in xrange(1, index_pointer.shape[0]):
            value = index_pointer[x] - index_pointer[x - 1]
            new_index_pointer.append(value)
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
        for index, each_file in enumerate(list_of_files):
            self.log.info("Iteration : {}".format(index))
            fv = hkl.load(each_file)
            partial_index_pointer = self.stats(fv)
            col_wise_dist = self.sum_up(partial_index_pointer, col_wise_dist)
            del fv
        return col_wise_dist

    def main(self):
        start_time = time()
        self.log.info("Generating column wise count of non zero elements")
        feature_vector_path = self.config['dimensionality_reduction']['feature_vector_path']
        feature_vector = self.helper.get_files_starts_with_extension("feature_vector_part-", feature_vector_path)
        self.log.info("Total number of files : {}".format(len(feature_vector)))
        col_wise_dist = self.get_stats(feature_vector)
        hkl.dump(np.asarray(col_wise_dist), open("col_wise_dist.dump", "w"))
        self.log.info("Total time for execution : {}".format(time() - start_time))


if __name__ == '__main__':
    stats = DataStats()
    stats.main()
