import math
import pickle as pi

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

from Clustering import KMeansImpl


class PairwiseJaccardSimilarity:
    def __init__(self):
        self.kmeans = KMeansImpl.KMeansImpl()

    @staticmethod
    def euclidean(a, b):
        temp_list = list()
        for x in xrange(len(a)):
            temp_list.append((a[x] - b[x]) ** 2)
        return math.sqrt(sum(temp_list))

    @staticmethod
    def jaccard_sim(input_matrix):
        mat = scipy.sparse.csr_matrix(input_matrix)
        cols_sum = mat.getnnz(axis=0)
        ab = mat.T * mat

        # for rows
        aa = np.repeat(cols_sum, ab.getnnz(axis=0))
        # for columns
        bb = cols_sum[ab.indices]

        similarities = ab.copy()
        similarities.data /= aa + bb - ab.data

        return similarities

    @staticmethod
    def pairwise_jaccard(X):
        """
        Computes the Jaccard distance between the rows of `X`.
        """
        intersect = X.dot(X.T)
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

    def main(self):
        input_matrix = pi.load(open("/home/satwik/Documents/Research/MachineLearning/LocalitySensitiveHashing/inp_100.dump"))
        names = pi.load(open("/home/satwik/Documents/Research/MachineLearning/LocalitySensitiveHashing/names_100.dump"))
        sim, dist = self.pairwise_jaccard(input_matrix)
        plt.hist(dist)
        plt.show()
        print self.kmeans.kmeans_pyspark(input_matrix, names, 9)
        # clusters = self.get_clusters(sim, 0.9)
        # print clusters


if __name__ == "__main__":
    pjs = PairwiseJaccardSimilarity()
    pjs.main()
