import pickle as pi
from time import time

from sklearn.decomposition.pca import PCA
from sklearn.decomposition.kernel_pca import KernelPCA

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from HelperFunctions.HelperFunction import HelperFunction
from PrepareData.LoadData import LoadData
from Clustering.DBScanClustering import DBScanClustering
from Clustering.HDBScanClustering import HDBScanClustering


class PrincipalComponentAnanlysis:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.load_data = LoadData()
        self.dbscan = DBScanClustering()
        self.hdbscan = HDBScanClustering()
        self.helper = HelperFunction()

    def pca(self, input_matrix, k_value, randomized=False):
        """
        Performs PCA. Here SVD will be solved using randomized algorithms.
        Function is not necessary explicitly as pca has svd_solver set to 'auto' by default which for higher dimension data performs randomized SVD.
        :param input_matrix: The input matrix ndarray form.
        :param k_value: The number of principal components to keep.
        :param randomized: Perform randomized svd under the hood if true.
        Disabled by default.
        :return:
        """
        self.log.info("Entering the {} class".format(self.pca.__name__))
        if randomized:
            pca = PCA(n_components=k_value, svd_solver='randomized')
        else:
            pca = PCA(n_components=k_value)
        reduced_matrix = pca.fit_transform(input_matrix)
        self.log.info("Exiting the {} class".format(self.pca.__name__))
        return reduced_matrix

    def kernel_pca(self, input_matrix, k_value):
        """
        Performs the Kernel Principal Component Reduction.
        :param input_matrix:
        :param k_value:
        :return:
        """
        self.log.info("Inside the {} class".format(self.pca.__name__))
        kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10, n_components=k_value)
        X_kpca = kpca.fit_transform(input_matrix)
        X_back = kpca.inverse_transform(X_kpca)
        self.log.info("Exiting the {} class".format(self.pca.__name__))
        return X_back

    def main(self):
        """
        The main method.
        :return:
        """
        pca_model_path = self.config["models"]["pca"]
        start_time = time()
        num_rows = 25000
        input_matrix, input_matrix_indices = self.load_data.main(num_rows=num_rows)
        # 1000 because we analyzed for various num_rows ranging from 100 to 25000.
        # We found 1st 1000 principal components are more than enough.
        reduced_matrix = self.pca(input_matrix.toarray(), 1000, randomized=True)
        self.log.info("Saving the model at : {}".format(pca_model_path))
        pi.dump(reduced_matrix, open("pca_reduced_matrix_" + str(num_rows) + ".model", "w"))
        eps_list = self.helper.frange(0.1, 1.0, 0.1)
        min_samples_list = range(2, 20, 2)
        min_cluster_size_list = range(2, 20, 2)

        dbscan_accuracy_params = self.dbscan.dbscan_cluster(input_matrix=reduced_matrix,
                                                            input_matrix_indices=input_matrix_indices,
                                                            eps_list=eps_list,
                                                            min_samples_list=min_samples_list)

        hdbscan_accuracy_params = self.hdbscan.hdbscan_cluster(input_matrix=reduced_matrix,
                                                               input_matrix_indices=input_matrix_indices,
                                                               min_cluster_size_list=min_cluster_size_list)

        self.log.info("DBScan Accuracy : {}".format(dbscan_accuracy_params))
        self.log.info("HDBScan Accuracy : {}".format(hdbscan_accuracy_params))
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    pca = PrincipalComponentAnanlysis()
    pca.main()
