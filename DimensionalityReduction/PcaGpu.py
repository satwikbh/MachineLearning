import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
import time
import numpy as np
import hickle
import os

from scipy.sparse import vstack, csr_matrix

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from HelperFunctions.HelperFunction import HelperFunction


class PcaGpu:
    def __init__(self):
        linalg.init()
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.config = ConfigUtil().get_config_instance()

    @staticmethod
    def transform_data_to_chunks(each_file, mini_batch_size):
        fv_list = []
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
            threshold_point = self.helper.get_threshold_point(s_gpu.get(), 0.9)
            self.log.info("Threshold point for the SIGMA matrix : {}".format(threshold_point))
            final__sigma_matrix.append(s_gpu.get().tolist())
            final_vt_matrix.append(vt_gpu.get()[:threshold_point, :])
            self.log.info("Shape of SIGMA matrix : {}\tShape of VT matrix : {}".format(s_gpu.shape,
                                                                                       vt_gpu[:threshold_point,
                                                                                       :].shape))
            self.log.info(
                "Time taken for svd computation of iteration number {} is : {}".format(index, time.time() - start_time))
        return fv_list, final__sigma_matrix, final_vt_matrix

    def store_matrix_to_disk(self, fv_list, final__sigma_matrix, final_vt_matrix, projected_matrix_full_path):
        self.log.info("Saving partial projected matrix to disk")
        final_input_matrix = vstack(fv_list)
        # Convert to sparse matrix ... done for saving disk space
        final_vt_matrix = [csr_matrix(x) for x in final_vt_matrix]
        final_projected_matrix = vstack(final_vt_matrix)

        hickle.dump(final_projected_matrix, open(projected_matrix_full_path, 'w'))
        self.log.info(
            "Input submatrix : {}\tProjected submatrix : {}".format(final_input_matrix.shape,
                                                                    final_projected_matrix.shape))

    def check_already_completed(self, feature_vector_path, projected_matrix_path):
        feature_vector = self.helper.get_files_starts_with_extension("feature_vector_part-", feature_vector_path)
        projected_matrix = self.helper.get_files_starts_with_extension("projected_matrix_part_", projected_matrix_path)
        projected_matrix_files = [x.split("projected_matrix_part_")[1] for x in projected_matrix]
        temp = list()

        for each_file in feature_vector:
            part = each_file.split("feature_vector_part-")[1]
            if part in projected_matrix_files:
                temp.append(each_file)

        for x in feature_vector:
            if x in temp:
                feature_vector.remove(x)

        return feature_vector

    def main(self):
        start_time = time.time()
        mini_batch_size = self.config['dimensionality_reduction']['mini_batch_size']
        feature_vector_path = self.config['dimensionality_reduction']['feature_vector_path']
        reduced_matrix_path = self.config['dimensionality_reduction']['reduced_matrix_path']
        projected_matrix_path = self.config['dimensionality_reduction']['projected_matrix_path']

        gpus_to_use = self.config['environment']['gpu_id']
        gpu_id = ','.join(gpus_to_use)
        self.log.info("Using the following gpus : {}".format(gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        # list_of_files = self.helper.get_files_starts_with_extension("feature_vector_part_", feature_vector_path)
        list_of_files = self.check_already_completed(feature_vector_path, projected_matrix_path)
        self.helper.create_dir_if_absent(reduced_matrix_path)
        self.helper.create_dir_if_absent(projected_matrix_path)
        meta_fv_list = list()

        for index, each_file in enumerate(list_of_files):
            projected_matrix_full_path = self.helper.get_full_path(projected_matrix_path,
                                                                   "projected_matrix_part_" + str(index) + ".hkl")
            fv_list = self.transform_data_to_chunks(each_file, mini_batch_size)
            fv_list, final__sigma_matrix, final_vt_matrix = self.compute_svd(fv_list)
            self.store_matrix_to_disk(fv_list, final__sigma_matrix, final_vt_matrix,
                                      projected_matrix_full_path)
            meta_fv_list += fv_list

        list_of_projected_matrix_path = self.helper.get_files_starts_with_extension(projected_matrix_path,
                                                                                    "projected_matrix_part_")
        final_input_matrix = np.vstack(meta_fv_list)
        final_projected_matrix = np.vstack([hickle.load(x) for x in list_of_projected_matrix_path])

        final_reduced_matrix = final_input_matrix.dot(final_projected_matrix.T)
        self.log.info("Final Reduced Matrix Shape : {}".format(final_reduced_matrix.shape))

        self.log.info("Total time taken : {}".format(time.time() - start_time))


if __name__ == '__main__':
    pca_gpu = PcaGpu()
    pca_gpu.main()
