from Miscellaneous.PairwiseDistanceMeasures import PairwiseDistanceMeasures
from Utils.LoggerUtil import LoggerUtil
from Utils.DBUtils import DBUtils
from Clustering.KMeansImpl import KMeansImpl
from Clustering.DBScanClustering import DBScan
from DimensionalityReduction.ClusteringMalwareIntoFamilies import SVD
from HelperFunctions.ParsingLogic import ParsingLogic

from leven import levenshtein

import urllib
import pickle as pi
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Driver:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.db_utils = DBUtils()
        self.distance_measures = PairwiseDistanceMeasures()
        self.kmeans = KMeansImpl()
        self.svd = SVD()
        self.dbscan = DBScan()
        self.parser = ParsingLogic()

    def main(self):
        username = "admin"
        password = urllib.quote("goodDevelopers@123")
        address = "localhost"
        port = "27017"
        auth_db = "admin"

        client = self.db_utils.get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=False,
                                          username=None, password=None)

        db = client['cuckoo']
        coll = db['clusteredMalware']

        query = {"key": {'$exists': True}}
        list_of_docs = coll.find(query).distinct("key")
        pi.dump(list_of_docs, open("names.dump", "w"))

        doc2bow = self.parser.parse_each_document(list_of_docs, coll)
        hcp = self.parser.convert2vec(doc2bow)
        input_matrix = np.array(hcp)
        pi.dump(input_matrix, open("input_matrix.dump", "w"))

        # input_file = pi.load(open("/home/satwik/Documents/Research/MachineLearning/Miscellaneous/input_matrix.dump"))

        # input_matrix = list(list())
        # for each in input_file:
        #     input_matrix.append(str(each))

        # input_matrix = np.array(input_matrix, dtype=np.str)

        labels = self.dbscan.dbscan_cluster(input_matrix, 0.5, metric='levenshtein')
        # labels = self.dbscan.dbscan_cluster(input_file, 0.5, metric=lambda x, y: levenshtein(input_file[int(x[0])],
        #                                                                                      input_file[int(y[0])]))

        self.log.info("DBScan on input_matrix(High Dimensional Data) : {}".format(labels))
        names = np.array(list_of_docs)

        U, SIGMA, VT = self.svd.singular_value_decomposition(input_matrix)
        threshold_point = self.svd.get_threshold_point(SIGMA)
        reduced_matrix = self.svd.perform_incremental_pca(input_matrix, threshold_point)

        labels = self.dbscan.dbscan_cluster(reduced_matrix, 0.5)
        self.log.info("DBScan on reduced_matrix(Low Dimensional Data) : {}".format(labels))

        print(self.kmeans.get_clusters_kmeans(input_matrix, names, k=10))

        self.log.info("************ Distance Matrix Calculation Ends *************")


if __name__ == '__main__':
    driver = Driver()
    driver.main()
