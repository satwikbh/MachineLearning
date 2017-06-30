from scipy.spatial import distance

import matplotlib

matplotlib.use('Agg')

import numpy as np
import time
from Clustering import KMeansImpl
from Utils.DBUtils import DBUtils
from Utils.LoggerUtil import LoggerUtil
from HelperFunctions.ParsingLogic import ParsingLogic

from sklearn.neighbors import DistanceMetric
from leven import levenshtein


class PairwiseDistanceMeasures:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.dist = DistanceMetric
        self.kmeans = KMeansImpl.KMeansImpl()
        self.db_utils = DBUtils()
        self.parser = ParsingLogic()

    def generalized_pairwise_distance_measure(self, input_matrix, metric="euclidean"):
        """
        Metric should be one of euclidean, hamming, jaccard, wminkowski, minkowski, mahalanobis,
        manhattan, kulsinski.
        The default value is euclidean.
        :param input_matrix:
        :param metric:
        :return:
        """
        self.log.info("************ Pairwise {} Distance Calculation Starts *************".format(metric))
        start_time = time.time()
        dist = self.dist.get_metric(metric)
        dist_matrix = dist.pairwise(input_matrix)
        self.log.info("************ Pairwise {} Distance Calculation Ends *************".format(metric))
        self.log.info("Time taken : {}".format(time.time() - start_time))
        return dist_matrix

    @staticmethod
    def pairwise_hamming(input_matrix):
        """
        Computes the Pairwise Hamming distance between the rows of `X`.
        :param input_matrix: The input matrix in numpy format
        :return:
        """
        dist_matrix = list(list())
        row_size = input_matrix.shape[0]
        for i in xrange(row_size):
            row = input_matrix[i]
            temp = list()
            for j in xrange(row_size):
                temp.append(distance.hamming(row, input_matrix[j]))
            dist_matrix.append(temp)
        return dist_matrix

    @staticmethod
    def pairwise_cosine(input_matrix):
        """
        Computes the Pairwise Cosine distance between the rows of `X`.
        :param input_matrix: The input matrix in numpy format
        :return:
        """
        # TODO: Can this method be improved just like pairwise_jaccard
        dist_matrix = list(list())
        row_size = input_matrix.shape[0]
        for i in xrange(row_size):
            row = input_matrix[i]
            temp = list()
            for j in xrange(row_size):
                temp.append(distance.cosine(row, input_matrix[j]))
            dist_matrix.append(temp)
        return dist_matrix

    @staticmethod
    def pairwise_euclidean(input_matrix):
        """
        Computes the Pairwise Euclidean distance between the rows of `X`.
        :param input_matrix: The input matrix in numpy format
        :return:
        """
        dist_matrix = list(list())
        row_size = input_matrix.shape[0]
        for i in xrange(row_size):
            row = input_matrix[i]
            temp = list()
            for j in xrange(row_size):
                temp.append(distance.euclidean(row, input_matrix[j]))
            dist_matrix.append(temp)
        return dist_matrix

    @staticmethod
    def pairwise_jaccard(input_matrix):
        """
        Computes the Pairwise Jaccard distance between the rows of `X`.
        :param input_matrix: Input Matrix in numpy format
        :return:
        """
        intersect = input_matrix.dot(input_matrix.T)
        row_sums = intersect.diagonal()
        unions = row_sums[:, None] + row_sums - intersect
        sim = np.asarray(intersect).__truediv__(unions)
        # sim = intersect / unions
        dist = 1 - sim
        return sim, dist

    @staticmethod
    def get_clusters(similarity_matrix, threshold):
        clusters = dict()
        count = 0
        for row in similarity_matrix:
            try:
                row_list = row.tolist()
                indices = [i for i, x in enumerate(row_list) if x > threshold]
                if indices not in clusters.values():  # and clusters.values().index(/home/satwikindices) != row_number:
                    clusters[count] = indices
                    count += 1
            except Exception as e:
                print(e)
        return clusters

    @staticmethod
    def lev_dist(x, y, input_file):
        return levenshtein(input_file[int(x[0])], input_file[int(y[0])])
