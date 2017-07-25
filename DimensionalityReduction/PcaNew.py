import pickle as pi
import numpy as np
import hickle

from scipy.linalg import svd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from Utils.LoggerUtil import LoggerUtil


class PcaNew:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    def get_threshold_point(self, sigma):
        """
        Takes the Eigen Values and then computes the position where 90% of the data is captured.
        :param sigma:
        :return:
        """
        self.log.info("************ Determining cut-off point *************")
        threshold_point = 0
        threshold = sum(sigma) * 0.9
        for x in xrange(len(sigma)):
            if sum(sigma[:x]) > threshold:
                threshold_point = x
                break
        self.log.info("The threshold point is : {}".format(threshold_point))
        return threshold_point

    def singular_value_decomposition(self, input_matrix):
        """
        Performs Singular Value Decomposition and return U, Sigma and VT matrices.
        :param input_matrix:
        :return:
        """
        self.log.info("************ Computing SVD starts *************")
        U, SIGMA, VT = svd(input_matrix, compute_uv=True)
        # pi.dump(U, open("U_matrix.dump", "w"))
        # pi.dump(SIGMA, open("Sigma_matrix.dump", "w"))
        # pi.dump(VT, open("VT_matrix.dump", "w"))
        self.log.info("************ Computing SVD ends *************")
        return U, SIGMA, VT

    def perform_incremental_pca(self, input_matrix, threshold_point):
        """
        An incremental version where the partial_fit method is called many times.
        This will provide a different slice of the dataset each time.
        Then performs PCA by reducing matrix to a dimension where 90% of the data is captured (till the threshold point)
        :param input_matrix:
        :param threshold_point:
        :return:
        """
        self.log.info("************ Computing Incremental PCA starts *************")
        ipca = IncrementalPCA(n_components=threshold_point)
        ipca.partial_fit(input_matrix)
        reduced_matrix = ipca.transform(input_matrix)
        pi.dump(reduced_matrix, open("Reduced_matrix.dump", "w"))
        self.log.info("************ Computing Incremental PCA ends *************")
        return reduced_matrix

    def dimensionality_reduction(self, input_matrix):
        input_matrix_std = StandardScaler().fit_transform(input_matrix)
        U, SIGMA, VT = self.singular_value_decomposition(input_matrix_std.T)
        threshold_point = self.get_threshold_point(SIGMA)
        projection_matrix = U[:, :threshold_point]
        reduced_matrix = input_matrix_std.dot(projection_matrix)
        # reduced_matrix = self.perform_incremental_pca(input_matrix, threshold_point)
        self.log.info("Reduced Matrix Shape : {}".format(reduced_matrix.shape))
        return reduced_matrix

    def prepare_data_for_pca(self, m, fv_path_name):
        self.log.info("Starting PCA")
        # Todo : Center the data ? How do you do this for the batches ?
        i_pca = IncrementalPCA(n_components=m)
        partial_matrix = hickle.load(open(fv_path_name))
        numpy_dense_array = np.asarray(partial_matrix.todense())
        i_pca.partial_fit(numpy_dense_array)

        x_transformed = None
        partial_matrix = hickle.load(open(fv_path_name))
        numpy_dense_array = np.asarray(partial_matrix.todense())
        x_chunk = i_pca.transform(numpy_dense_array)
        if x_transformed is None:
            x_transformed = x_chunk
        else:
            x_transformed = np.vstack((x_transformed, x_chunk))

        return x_transformed
