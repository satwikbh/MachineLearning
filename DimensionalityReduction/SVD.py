import json
import logging.config
import os
import pickle as pi
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix


class SVD:
    logger = logging.getLogger(__name__)

    def __init__(self):
        pass

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
    def load_data():
        X = sparse_random_matrix(100, 100, density=0.01, random_state=42)
        return X

    @staticmethod
    def load_data1():
        """
        This will load the data from mycsvfile.csv and convert it into Numpy Array.
        :return:
        """
        f = open("mycsvfile.csv")
        l = list(list())
        names = list()

        for lines in f.readlines():
            split = lines.split(",")
            names.append(split[0])
            l.append(list(split[1][:-2]))

        return np.array(l).transpose()

    def main(self):
        start_time = time.time()
        SVD.logger.info("Entering the {} class".format(SVD.main.__name__))
        input_matrix = self.load_data()
        SVD.logger.info("Matrix Shape {} ".format(input_matrix.shape))
        print("Enter the number of dimensions : ")
        n_components = int(raw_input())
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(input_matrix)
        pi.dump(svd.explained_variance_ratio_, open("Eigen_Values.dump", "w"))
        SVD.logger.info("SVD Variance Ratio : {} ".format(svd.explained_variance_ratio_))
        SVD.logger.info("SVD Variance Sum : {} ".format(svd.explained_variance_ratio_.sum()))
        SVD.logger.info("Total time taken : {} ".format(time.time() - start_time))
        SVD.logger.info("Exiting the {} class".format(SVD.main.__name__))


if __name__ == '__main__':
    svd = SVD()
    svd.setup_logging()
    svd.main()

