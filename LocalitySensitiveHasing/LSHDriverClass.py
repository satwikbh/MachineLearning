import pickle as pi
import time
import urllib

import numpy as np

import HelperFunctions.helper_functions as utils
import Utils.DBUtils as dbUtils
from Clustering.SingleLinkageClustering import SingleLinkageClustering
from DatasketchLsh import DatasketchLsh
from FalconnLsh import FalconnLsh
from NearpyLsh import NearpyLsh
from ParsingLogic import ParsingLogic
from Utils import LoggerUtil


class DriverClass:
    def __init__(self):

        self.log = LoggerUtil.LoggerUtil(self.__class__).get()

        self.parser = ParsingLogic()
        self.falconn = FalconnLsh()
        self.datasketch = DatasketchLsh()
        self.nearpy = NearpyLsh()
        self.slc = SingleLinkageClustering()

    def main(self, num_variants):
        username = "admin"
        password = urllib.quote("goodDevelopers@123")
        address = "localhost"
        port = "27017"
        auth_db = "admin"

        client = dbUtils.DBUtils.get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=False,
                                            username=None, password=None)

        # client = dbUtils.DBUtils.get_client(address, port, username, password, auth_db, is_auth_enabled=True)

        db = client['cuckoo']
        coll = db['cluster2db']

        query = {"key": {'$exists': True}}
        list_of_docs = coll.find(query).distinct("key")

        assert num_variants < len(list_of_docs)

        if utils.HelperFunction.is_file_present("inp_" + str(num_variants) + ".dump"):
            inp = pi.load(open("inp_" + str(num_variants) + ".dump"))
        else:
            self.parser.parse_each_document(list_of_docs[:num_variants], coll)
            hcp = self.parser.convert2vec()
            inp = np.array(hcp)
            pi.dump(inp, open("inp_" + str(num_variants) + ".dump", 'w'))

        threshold = 0.7

        clusters_falconn = self.falconn.lsh_falconn(inp, threshold=threshold)
        clusters_datasketch = self.datasketch.lsh_datasketch(inp, threshold, inp.shape[1])
        clusters_nearpy_lsh = self.nearpy.nearpy_lsh(inp)

        pi.dump(clusters_falconn, open("clusters_falconn_" + str(num_variants) + ".dump", "w"))
        pi.dump(clusters_datasketch, open("clusters_datasketch_" + str(num_variants) + ".dump", "w"))
        pi.dump(clusters_nearpy_lsh, open("clusters_nearpy_lsh_" + str(num_variants) + ".dump", "w"))

        self.slc.single_linkage_clustering(inp, threshold, num_variants)


if __name__ == '__main__':
    start_time = time.time()

    driver = DriverClass()
    num_variants = 10
    driver.main(num_variants)

    driver.log.info("Overall time taken : {}".format(time.time() - start_time))
