import json
import logging.config
import os
import pickle as pi
import time

import ipdb
import matplotlib
import numpy as np
from sklearn import manifold
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import kernel_pca as kernelPCA

matplotlib.use('Agg')

import matplotlib.pyplot as plt


class PCA:
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.logger.info("Initialized the class")

    @staticmethod
    def setup_logging(default_path='logging.json', default_level=logging.INFO, env_key='LOG_CFG'):
        """
        Setup logging configuration
        :param default_path:
        :param default_level:
        :param env_key:
        :return:
        """
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)

    @staticmethod
    def plotting(value, name):
        ipdb.set_trace()  # debugging starts here
        plt.plot(np.array(value[:, 0]), np.array(value[:, 1]), "ro")
        plt.savefig(name + ".png")

    @staticmethod
    def randomized_pca(input_matrix, k_value):
        """
        Performs PCA. Here SVD will be solved using randomized algorithms.
        Function is not necessary explicitly as pca has svd_solver set to 'auto' by default which for higher dimension data performs randomized SVD.
        :param input_matrix:
        :param k_value:
        :return:
        """
        PCA.logger.info("Entering the {} class".format(PCA.randomized_pca.__name__))
        pca = sklearnPCA(n_components=k_value, svd_solver='randomized')
        pca.fit_transform(input_matrix)
        value = pca.components_
        PCA.logger.info(value.transpose())
        PCA.logger.info(value.shape)
        PCA.plotting(value.transpose(), PCA.randomized_pca.__name__)
        PCA.logger.info("Exiting the {} class".format(PCA.randomized_pca.__name__))

    @staticmethod
    def pca(input_matrix, k_value):
        """
        Performs the Principal Component Reduction.
        :param input_matrix:
        :param k_value:
        :return:
        """
        PCA.logger.info("Inside the {} class".format(PCA.pca.__name__))
        sklearn_pca = sklearnPCA(n_components=k_value)
        sklearn_pca.fit_transform(input_matrix)
        value = sklearn_pca.components_
        pi.dump(sklearn_pca.explained_variance_ratio_, open("Eigen_Values.dump", "w"))
        PCA.logger.info("PCA Variance sum : {}".format(sklearn_pca.explained_variance_ratio_.sum()))
        PCA.logger.info(value.transpose())
        PCA.logger.info(value.shape)
        PCA.plotting(value.transpose(), PCA.pca.__name__)
        PCA.logger.info("Exiting the {} class".format(PCA.pca.__name__))

    @staticmethod
    def kernel_pca(input_matrix, k_value):
        """
        Performs the Kernel Principal Component Reduction.
        :param input_matrix:
        :param k_value:
        :return:
        """
        PCA.logger.info("Inside the {} class".format(PCA.kernel_pca.__name__))
        kpca = kernelPCA.KernelPCA(n_components=k_value, fit_inverse_transform=True)
        X_kpca = kpca.fit_transform(input_matrix)
        sklearn_pca = sklearnPCA(n_components=k_value)
        X_pca = sklearn_pca.fit_transform(input_matrix)
        PCA.logger.info(X_pca.shape, X_pca)
        PCA.logger.info(X_kpca.shape, X_kpca)
        # PCA.plotting(value.transpose())
        PCA.logger.info("Exiting the {} class".format(PCA.kernel_pca.__name__))

    @staticmethod
    def lle(input_matrix, k_value):
        """
        Performs Local Linear Embedding.
        :param input_matrix:
        :param k_value:
        :return:
        """
        n_neighbors = 30
        PCA.logger.info("Inside the {} class".format(PCA.lle.__name__))
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=k_value, method='standard')
        x_lle = clf.fit_transform(input_matrix)
        PCA.logger.info("Value of lle : {}".format(x_lle))
        PCA.plotting(x_lle, PCA.lle.__name__)
        PCA.logger.info("Exiting the {} class".format(PCA.lle.__name__))

    @staticmethod
    def t_sne(input_matrix, k_value):
        """
        Performs tsne.
        :param input_matrix:
        :param k_value:
        :return:
        """
        PCA.logger.info("Entering the {} method".format(PCA.t_sne.__name__))
        tsne = manifold.TSNE(n_components=k_value, init='pca', random_state=0)
        x_tsne = tsne.fit_transform(input_matrix)
        PCA.logger.info("Value of tsne : {}".format(x_tsne))
        PCA.plotting(x_tsne.transpose(), PCA.t_sne.__name__)
        PCA.logger.info("Exiting the {} method".format(PCA.t_sne.__name__))

    @staticmethod
    def switch(reduction_function, input_matrix, k_value):
        """
        This function will act as a switch case.
        :param reduction_function:
        :param input_matrix:
        :param k_value:
        :return:
        """
        try:
            if reduction_function == 1:
                PCA.pca(input_matrix, k_value)
            if reduction_function == 2:
                PCA.randomized_pca(input_matrix, k_value)
            if reduction_function == 3:
                PCA.kernel_pca(input_matrix, k_value)
            if reduction_function == 4:
                PCA.lle(input_matrix, k_value)
            if reduction_function == 5:
                PCA.t_sne(input_matrix, k_value)
        except Exception as e:
            PCA.logger.error("Only three states supported. \n The error is {} ".format(e))

    @staticmethod
    def prepare_data():
        """
        Prepares the data for the processing.
        :return:
        """
        PCA.logger.info("Inside the {} class".format(PCA.prepare_data.__name__))
        f = open("mycsvfile.csv")
        l = list(list())
        names = list()

        for lines in f.readlines():
            split = lines.split(",")
            names.append(split[0])
            l.append(list(split[1][:-2]))

        input_matrix = np.array(l).transpose()
        PCA.logger.info("Shape of the Matrix is : {}".format(input_matrix.shape))
        PCA.logger.info("Exiting the {} class".format(PCA.prepare_data.__name__))
        return input_matrix

    @staticmethod
    def main():
        """
        The main method.
        :return:
        """
        PCA.logger.info("Inside the {} class".format(PCA.main.__name__))
        input_matrix = PCA.prepare_data()
        print(
            "Enter the choice of reduction \n "
            "1. Simple PCA \n "
            "2. PCA + Randomized SVD \n "
            "3. Kernel PCA \n "
            "4. LLE \n "
            "5. t-SNE"
        )
        reduction_function = int(raw_input())
        print("Enter the dimensions to which you want to reduce ")
        k_value = int(raw_input())
        PCA.switch(reduction_function, input_matrix, k_value)
        PCA.logger.info("Exiting the {} class".format(PCA.main.__name__))


if __name__ == '__main__':
    start_time = time.time()
    pca = PCA()
    pca.setup_logging()
    pca.main()
    logging.info("Total time taken : {} ".format(time.time() - start_time))
