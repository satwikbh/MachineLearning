import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
import time
import numpy as np
import hickle

from scipy.sparse import vstack

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from HelperFunctions.HelperFunction import HelperFunction


class PcaGpu:
    def __init__(self):
        linalg.init()
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.config = ConfigUtil().get_config_instance()['dimensionality_reduction']

    @staticmethod
    def transform_data_to_chunks(batch_of_files, mini_batch_size):
        fv_list = []
        for each_file in batch_of_files:
            fv = hickle.load(open(each_file)).astype(np.float32)
            for x in xrange(fv.shape[0] / mini_batch_size):
                fv_list.append(fv[x:x + mini_batch_size])
        return fv_list

    def compute_svd(self, fv_list):
        final__sigma_matrix = list()
        final_vt_matrix = list()

        for index, sub_matrix in enumerate(fv_list):
            start_time = time.time()
            sub_matrix_gpu = gpuarray.to_gpu(sub_matrix.todense())
            s_gpu, vt_gpu = linalg.svd(sub_matrix_gpu, 'N', 'S', lib="cusolver")
            final__sigma_matrix.append(s_gpu.get().tolist())
            final_vt_matrix.append(vt_gpu.get())
            self.log.info("Shape of SIGMA matrix : {}\tShape of VT matrix : {}".format(s_gpu.shape, vt_gpu.shape))
            threshold_point = self.helper.get_threshold_point(s_gpu, 0.9)
            self.log.info("Threshold point for the SIGMA matrix : {}".format(threshold_point))
            self.log.info(
                "Time taken for svd computation of iteration number {} is : {}".format(index, time.time() - start_time))
        return fv_list, final__sigma_matrix, final_vt_matrix

    def reduce_matrix(self, fv_list, final__sigma_matrix, final_vt_matrix):
        self.log.info("Stacking the partial matrices")
        final_input_matrix = vstack(fv_list)
        final_projected_matrix = np.vstack(final_vt_matrix)

        final_reduced_matrix = final_input_matrix.dot(final_projected_matrix.T)
        self.log.info(
            "Input submatrix : {}\tProjected submatrix : {}\tReduced submatrix : {}".format(final_input_matrix.shape,
                                                                                            final_projected_matrix.shape,
                                                                                            final_reduced_matrix.shape))
        return final_reduced_matrix

    def main(self):
        start_time = time.time()
        mini_batch_size = self.config['mini_batch_size']
        files_per_batch = self.config['files_per_batch']
        feature_vector_path = self.config['feature_vector_path']
        reduced_matrix_path = self.config['reduced_matrix_path']
        list_of_files = self.helper.get_files_with_extension("feature_vector_", feature_vector_path)
        self.helper.create_dir_if_absent(reduced_matrix_path)

        for index, batch_of_files in enumerate(self.helper.batch(list_of_files, files_per_batch)):
            fv_list = self.transform_data_to_chunks(batch_of_files, mini_batch_size)
            fv_list, final__sigma_matrix, final_vt_matrix = self.compute_svd(fv_list)
            final_reduced_matrix = self.reduce_matrix(fv_list, final__sigma_matrix, final_vt_matrix)
            file_name = self.helper.get_full_path(reduced_matrix_path, "reduced_matrix_part_" + str(index) + ".hkl")
            hickle.dump(final_reduced_matrix, open(file_name, 'w'))
        self.log.info("Total time taken : {}".format(time.time() - start_time))


if __name__ == '__main__':
    pca_gpu = PcaGpu()
    pca_gpu.main()
