import pickle as pi
import time
import urllib

import numpy as np

import HelperFunctions.HelperFunction as utils
from Utils.DBUtils import DBUtils
from Clustering.BirchClustering import BirchClustering
from Clustering.DBScanClustering import DBScan
from Clustering.SingleLinkageClustering import SingleLinkageClustering
from DatasketchLsh import DatasketchLsh
from FalconnLsh import FalconnLsh
from HelperFunctions.ParsingLogic import ParsingLogic
from NearpyLsh import NearpyLsh
from Utils.LoggerUtil import LoggerUtil


class DriverClass:
    def __init__(self):

        self.log = LoggerUtil(self.__class__).get()
        self.db_utils = DBUtils()

        self.parser = ParsingLogic()
        self.falconn = FalconnLsh()
        self.datasketch = DatasketchLsh()
        self.nearpy = NearpyLsh()

        self.slc = SingleLinkageClustering()
        self.dbscan = DBScan()
        self.birch = BirchClustering()

    def main(self, num_variants):
        username = "admin"
        password = urllib.quote("goodDevelopers@123")
        address = "localhost"
        port = "27017"
        auth_db = "admin"

        client = self.db_utils.get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=False,
                                          username=None, password=None)

        # client = dbUtils.DBUtils.get_client(address, port, username, password, auth_db, is_auth_enabled=True)

        db = client['cuckoo']
        coll = db['cluster2db']

        query = {"key": {'$exists': True}}
        list_of_docs = coll.find(query).distinct("key")
        pi.dump(list_of_docs, open("names_" + str(num_variants) + ".dump", "w"))

        assert num_variants <= len(list_of_docs)

        # temp = [u'VirusShare_000116a8a82aa71872762d42e2a5befd', u'VirusShare_0001287646dfe387efa2532f708941c2',
        #         u'VirusShare_0001faece12c738423ce08f71936d6cc', u'VirusShare_000338a9a8a6118f1821c7ef9a4c2145',
        #         u'VirusShare_0005615843e0b79562d1b241d0134f52', u'VirusShare_000564b760fa1522652eb7120a8d6331',
        #         u'VirusShare_000702402b1440119ece10d7cfee7e34', u'VirusShare_000837415dfef310371e048c96163d96',
        #         u'VirusShare_000b97180665824e9d07ad6866c02438', u'VirusShare_000b97a31687866cf8c211a914bf7521',
        #         u'VirusShare_000c598af0846e8d914bb71b0950220e', u'VirusShare_000c62b19d0ebc708da67cfb8410569c',
        #         u'VirusShare_000c9197a80dcca2cfc0f1737c65d560', u'VirusShare_000c99771b0c9abbba882fc921a68458',
        #         u'VirusShare_000cc24161a644f11ad41c051a3f5f04', u'VirusShare_000ccfc19a4b7152dce2cba37ef6b867',
        #         u'VirusShare_000cd266c7da16f0f1b47cfdae1702c9', u'VirusShare_000d5dcbad631c6c24f607f65b272986',
        #         u'VirusShare_000e4a2f2f53e31a06c4f128dc0e7aea', u'VirusShare_000f0a671fb47e91e34f0cd6b0bff639',
        #         u'VirusShare_00106797b9eb144a299afbd1335d33ff', u'VirusShare_0011f3c9b9d8dab94cb0109c14765ed0',
        #         u'VirusShare_001295727995d65f4da809b30758fa7b', u'VirusShare_0012fc946049bec0eff294a736b8a36f',
        #         u'VirusShare_0013f027d3579bfbc8b3b9c8bb1deb47', u'VirusShare_001417ead59ec0025ad7bcf3a7036d36',
        #         u'VirusShare_001561c35a740f9f6cfa5bacbdc091ba', u'VirusShare_001992c8c9fd8c29d250ebf7ff48c9d6',
        #         u'VirusShare_001a782949ffddb73ba77c90bea7b7b5', u'VirusShare_001a987d9a7837340eaaf8e75cea3bdb',
        #         u'VirusShare_001aeb0a58f984397b5161454d3c292f', u'VirusShare_001be2a45e6960b66a478c53b0dc9ed9',
        #         u'VirusShare_001bedcbcf832338b9796e99b653050f', u'VirusShare_001bef215cf10b580aad306381e1de08',
        #         u'VirusShare_001c222e4fa8cd7d28fe01c222e21d00', u'VirusShare_001d4c832fc774e1bfc2852d5c9f421d',
        #         u'VirusShare_001e01440820ef6dcf13823bf0bcd2b1', u'VirusShare_001eadb176d28c2f36312a83e0259eb9',
        #         u'VirusShare_001f3bd0970d177d5617e2b7f4308a3c', u'VirusShare_0021ae87ba45a8a6bcab3f92a4d08aa7',
        #         u'VirusShare_0021bffa0f0f021a99608c89ba3e2166', u'VirusShare_002493892a696937be1db62d7b1a2a46',
        #         u'VirusShare_00249c829c91d9d45627f76044903daf', u'VirusShare_002587e1d397b73f6c4328fc5feb3445',
        #         u'VirusShare_00288fca711a9af54672b88c2cb9d73a', u'VirusShare_00296c42b0ee1db5f900908fcaf0c9db',
        #         u'VirusShare_00296c79bbc1ef385052adc553869da2', u'VirusShare_002a2ff3ccf636040445b49866e2bea7',
        #         u'VirusShare_002ae419dd979124de09c62a9973c7d4', u'VirusShare_002b3e8ea03fe10f1d93939c7d15c806',
        #         u'VirusShare_002bca11ea496008bde45ebb4c740ab7', u'VirusShare_002c469059fdbbf2bdbb7fbf92f398ab',
        #         u'VirusShare_002d7a92463754a181dbbccb96828970', u'VirusShare_002da72be69fd1d3b352d5a3b971a77c',
        #         u'VirusShare_002dee4f23b7d34490a17e4f30c3eaf1', u'VirusShare_002eadd8a0c069c2d2cccc9554a8918d',
        #         u'VirusShare_002f336b791897bcd03dc3243f611e2e', u'VirusShare_002fdcb4e749f11e5547443231c22ac0',
        #         u'VirusShare_003120c3cb64ab32727d260b7866b127', u'VirusShare_00314428ed8509ec3a7e2e59aacf89c2',
        #         u'VirusShare_00315327d4dd71cd7b7db222b9e88037', u'VirusShare_0031c86f8309f38ea22afccddd22a0e1',
        #         u'VirusShare_0032487db3fbff178ff61b3c38d32ab1', u'VirusShare_0032dfe6d255df9dd00d696736d7bfa7',
        #         u'VirusShare_00334e1a54cd16a95b2a0efde295f292', u'VirusShare_00336021c8cb5e08c8510f5259dde2f8',
        #         u'VirusShare_0033ba03437486b43a2b35a955f944d3', u'VirusShare_003440c16a3a2f12efbaee92a828b5b9',
        #         u'VirusShare_003498ccb8fb4d3b9d01c6509a790d23', u'VirusShare_0034c23e801fb95d875b2eda9c682e18',
        #         u'VirusShare_00357dede65d73984d2bbadea012014d', u'VirusShare_0035923dfc0dfc460d8d49a77af2cf87',
        #         u'VirusShare_0037251f41ce7871ee45c1628c71cdd3', u'VirusShare_0037c964a6fe38b2532a0d31b1205ce2',
        #         u'VirusShare_0039c532a807d890cf5705b586c0fdf4', u'VirusShare_003a85f651adadfe5a4ec289ba27154f',
        #         u'VirusShare_003b3693236ffc9eabb8bf6388be53c2', u'VirusShare_003be593e8b6f72aa2494bdd1bc9ff82',
        #         u'VirusShare_003c801a81d05e82d2b85a0c045eb8d1', u'VirusShare_003c8fca7e874af41544a92254758a72',
        #         u'VirusShare_003ce5d8a90624f8c7f19fcfa6206b4b', u'VirusShare_003d282c867b2265986d17fa08ea7f71',
        #         u'VirusShare_003e03a72a2f6471a33cc6ef7e45979a', u'VirusShare_003e8bae51b1aeae281f0162564e9717',
        #         u'VirusShare_003ea5dfe8c8ca9ae2d51ed14d99be80', u'VirusShare_003fa644c8c30fbd50c0c50ca6fa3735',
        #         u'VirusShare_003fad729e4b793f8e85d03c43375187', u'VirusShare_003fd06dadc57862641f1137811f8ebc',
        #         u'VirusShare_00432cf920437852c70542083f39b7b5', u'VirusShare_0043581ba6e03639560db3ac7d5e1557',
        #         u'VirusShare_004514897de6c26963cf03ae4239f5b0', u'VirusShare_0047e7ddd53ce3e1627047eeaf6e11c3',
        #         u'VirusShare_004893a0067d2841a90326af6f81462a', u'VirusShare_004a58dfe83d756661df4986a04fbd56',
        #         u'VirusShare_004ad15c81faea7d284b9a67d60acdf1', u'VirusShare_004b2ff32a57e0749d6495c433a4494c',
        #         u'VirusShare_004bc3be7178dfcbdc061a3f4ef7b810', u'VirusShare_004c4150fc7420a84ed4b84ca570936b',
        #         u'VirusShare_004c80a601e4b7259c266abbd58416c7', u'VirusShare_004d2851a23cfdec2f1ec695f220c29b']

        if utils.HelperFunction.is_file_present("inp_" + str(num_variants) + ".dump"):
            inp = pi.load(open("inp_" + str(num_variants) + ".dump"))
        else:
            # self.parser.parse_each_document(list_of_docs[:num_variants], coll)
            doc2bow = self.parser.parse_each_document(list_of_docs, coll)
            hcp = self.parser.convert2vec(doc2bow)
            inp = np.array(hcp)
            pi.dump(inp, open("inp_" + str(num_variants) + ".dump", 'w'))

        threshold = 0.4

        clusters_falconn = self.falconn.lsh_falconn(inp, threshold=threshold)
        # clusters_datasketch = self.datasketch.lsh_datasketch(inp, threshold, inp.shape[1])
        # clusters_nearpy_lsh = self.nearpy.nearpy_lsh(inp)

        # pi.dump(clusters_falconn, open("clusters_falconn_" + str(num_variants) + ".dump", "w"))
        # pi.dump(clusters_datasketch, open("clusters_datasketch_" + str(num_variants) + ".dump", "w"))
        # pi.dump(clusters_nearpy_lsh, open("clusters_nearpy_lsh_" + str(num_variants) + ".dump", "w"))

        self.slc.single_linkage_clustering(inp, threshold, num_variants)
        # clusters_dbscan = self.dbscan.dbscan_cluster(input_matrix=inp, threshold=threshold)
        # pi.dump(clusters_dbscan, open("clusters_dbscan_" + str(num_variants) + ".dump", "w"))

        # clusters_birch = self.birch.birch_clustering(inp, threshold)
        # pi.dump(clusters_birch, open("clusters_birch_" + str(num_variants) + ".dump", "w"))


if __name__ == '__main__':
    start_time = time.time()

    driver = DriverClass()
    num_variants = 100
    driver.main(num_variants)

    driver.log.info("Overall time taken : {}".format(time.time() - start_time))
