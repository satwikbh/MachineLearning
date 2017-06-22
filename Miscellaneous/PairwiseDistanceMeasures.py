import math
import urllib
from scipy.spatial import distance

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pickle as pi

from Clustering import KMeansImpl
from Utils.DBUtils import DBUtils
from Utils.LoggerUtil import LoggerUtil
from HelperFunctions.ParsingLogic import ParsingLogic


class PairwiseDistanceMeasures:
    def __init__(self):
        self.kmeans = KMeansImpl.KMeansImpl()
        self.db_utils = DBUtils()
        self.parser = ParsingLogic()
        self.log = LoggerUtil(self.__class__.__name__).get()

    @staticmethod
    def pairwise_hamming(X):
        """
        Computes the Pairwise Hamming distance between the rows of `X`.
        :param X: Input Matrix in numpy format
        :return:
        """
        dist_matrix = list(list())
        x, y = X.shape
        for i in xrange(x):
            row = X[i]
            temp = list()
            for j in xrange(y):
                temp.append(distance.hamming(row, X[i][j]))
            dist_matrix.append(temp)
        return dist_matrix

    @staticmethod
    def pairwise_cosine(X):
        """
        Computes the Pairwise Cosine distance between the rows of `X`.
        :param X: Input Matrix in numpy format
        :return:
        """
        # TODO: Can this method be improved just like pairwise_jaccard
        dist_matrix = list(list())
        x, y = X.shape
        for i in xrange(x):
            row = X[i]
            temp = list()
            for j in xrange(y):
                temp.append(distance.cosine(row, X[i][j]))
            dist_matrix.append(temp)
        return dist_matrix

    @staticmethod
    def pairwise_euclidean(X):
        """
        Computes the Pairwise Euclidean distance between the rows of `X`.
        :param X: Input Matrix in numpy format
        :return:
        """
        dist_matrix = list(list())
        x, y = X.shape
        for i in xrange(x):
            row = X[i]
            temp = list()
            for j in xrange(y):
                temp.append(distance.euclidean(row, X[i][j]))
            dist_matrix.append(temp)
        return dist_matrix

    @staticmethod
    def pairwise_jaccard(X):
        """
        Computes the Pairwise Jaccard distance between the rows of `X`.
        :param X: Input Matrix in numpy format
        :return:
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
        # list_of_docs = coll.find(query).distinct("key")
        list_of_docs = ["VirusShare_30808b35ee44a152d22d895487a74499", "VirusShare_312e7b25ae0458b997a2d75c4b95b98f",
                        "VirusShare_3196b9ce12a6da0dcf4af2cb414ffda6", "VirusShare_320bdae17f7ac8563245de5a0d9a88d9",
                        "VirusShare_321a2827916080ed7a19a2f1925429de", "VirusShare_34654d47a1cd0b4c91f30ca407e964fc",
                        "VirusShare_36a5531e9ef4913da8dfbf640323ef2f", "VirusShare_36bdca732107a81b56a593ca69fbcdf2",
                        "VirusShare_38c6d09007e31339e0d3bafa523d5390", "VirusShare_38e5a3e0057161345cee08b20d5c315c",
                        "VirusShare_3a73647cfb7c5376567f7708e3fd156f", "VirusShare_3d72602ed94de1973e10b95fc88cb2de",
                        "VirusShare_3f8354169e6e50caeb943d337dd0291c", "VirusShare_71dce9051a7b455c3ebee189868ade60",
                        "VirusShare_72fa9495c9570587ec75ff71d6d482bc", "VirusShare_75633a3deadf11ee56c6ec02d8d624a6",
                        "VirusShare_76b75382227c68171f9d288a434aace5", "VirusShare_77f43b57555effe2bc066cbb2e1af6a0",
                        "VirusShare_78aacf3221ecb6a87e8ab56dc2ce3c80", "VirusShare_7999eaa11ee76d8f96253754b58eb5d4",
                        "VirusShare_79b46375bb449fd53bae5c1294031ce0", "VirusShare_7a681c002af362f3cb1c6c1c8016a5cc",
                        "VirusShare_7afacc1c2dac76dd77ab2e30582a8e3c", "VirusShare_b0fcf4f3cff08e7edafe5c0e61c78a53",
                        "VirusShare_b1286bb96d2e17e5e76ea21f8e09e631", "VirusShare_b18012d4fff9a3552e0b96ddcdbbd2fa",
                        "VirusShare_b18504088fb44bbec8e28a547ab07dc1", "VirusShare_b19ad6441b8f8cc3294e47ff55517274",
                        "VirusShare_b1c6a843418241763360f75e9952c1d6", "VirusShare_b3233adef9ecdaf34fb58067b6ec3a35",
                        "VirusShare_b34a60c019c6597cd550640e693cbff0", "VirusShare_b36886da50f68a09dc9faa34080d2622",
                        "VirusShare_b3e75c387e9175714d2b1d322be4d722", "VirusShare_b43321b02a6b057096487b218768334a",
                        "VirusShare_b483668366f9b736479649c067261188", "VirusShare_b4927ab47ef7b15324c694a5c6a940e7",
                        "VirusShare_b4a0708568ef511ca6dc6e32f9980e5e", "VirusShare_b4bb6716045060891f96c34ac200da38",
                        "VirusShare_b633cedce7a8f0a567e8f502cef0bf74", "VirusShare_b65766040bbb7d3b075bc4c8a4ad2a80",
                        "VirusShare_b664790913accea0db4b296c5776f98a", "VirusShare_b6c24574e8b5c8911f2f85dccb7693b0",
                        "VirusShare_b6cdbe79f24b6b5b8d8cb27788e809a3", "VirusShare_b6d84d314905c4e52ace3607883ad948",
                        "VirusShare_b83730e19880949537c991d95e455679", "VirusShare_ba86e2d2c06a97a580b0732a3e03af8d",
                        "VirusShare_babc13179690f123ae55accf45c6e84c", "VirusShare_bb70f25feb2882adcc4498b1490a019d",
                        "VirusShare_bb79935f11f93e6f58500db1ca50b938", "VirusShare_bb91f142b268b1daf77d39bdc7f938ed",
                        "VirusShare_bbbec337e7ff4c5a615a25ce9d9c94d5", "VirusShare_bbcf05a5f4c3ccd4a14c69a3364f6ea0",
                        "VirusShare_bbdd83a36098f13e9b517ba72cb35e7c", "VirusShare_bbe1f79cb05443e7f7707ed4dd9e2d85",
                        "VirusShare_bd0771706e3e2c3aa0e59ac0e95f287f", "VirusShare_bd07edd0ddae2194c13f1d4edd8fc206",
                        "VirusShare_bd2d040462fcfd4ac145ad350a1416f9", "VirusShare_bd4fcf014ec82fa15d2c960166b0858e",
                        "VirusShare_bd7bcfc3b9a2ae6aa855501bd425a444", "VirusShare_bd8eb1aad41c08973e2997bcbd75635e",
                        "VirusShare_be21d7015255c609b1362e5303b20857", "VirusShare_be4ca79a68d8954269035eb27ffd70a0",
                        "VirusShare_bea16fe3a312427f12225124a4429269", "VirusShare_beb3ffc342b8e393be3a9bbfcc648a61",
                        "VirusShare_bf1250b9a846d9d645cce1775f2b8d9b", "VirusShare_bf26947b61b1166f4ad1f8fa1ea8c638",
                        "VirusShare_bf35b376ded544f42c859bba478737e1", "VirusShare_bf5756fb046a39004273ef28db43a203",
                        "VirusShare_bfb156b6f09aeaf61b45be30aff4a160", "VirusShare_bfcceca88ee234d02412136e50e0cd23",
                        "VirusShare_bff406cb7aadcb2b8e9ed91d746412be", "VirusShare_e03650e1126df2158fb61f51e4b728a0",
                        "VirusShare_e03baac19bda313da6e0c746d286d571", "VirusShare_e04c82a48c09213b5533967e1f5d9cd0",
                        "VirusShare_e06271f1a3e7800a07f856ef504cbe2b", "VirusShare_e09a4e74b6e92d6582bbc92d62a6ef29",
                        "VirusShare_e0bb77f315dfde8c65f2e699f3b4b1a0", "VirusShare_e0e41077549753f0a0ee7f254536f93f",
                        "VirusShare_e0ec0ec1bfa9a9777d6a97ae7cffb78f", "VirusShare_e0f76fb95bbcccfb05af985015b48380",
                        "VirusShare_e1871d0ed23decff074418152e50a883", "VirusShare_e1bab2109ba72187d178dc52676914f3",
                        "VirusShare_e1c20f34431b696fd05e6e53cd31d030", "VirusShare_e1e6cd9b2e97415f03952a30ccc711e3",
                        "VirusShare_e2907acdac9876feb00280b0716070df", "VirusShare_e2aad246762217b228aa19dfe9a1b860",
                        "VirusShare_e2f52bb785618643770d0c9d0aebbdca", "VirusShare_e39cba64c526a9dc617f4bd4b27f89f2",
                        "VirusShare_e40b016ad87adaf3254e4ba40d222733", "VirusShare_f0026ff5be1f449baceb343e5e1333c8",
                        "VirusShare_f03f66b96aeb28dbbeda059646c36c8f", "VirusShare_f050d30a209a85a5a3f7660d061a5874",
                        "VirusShare_f09c846010d830469eb7d1b88aa8a514", "VirusShare_f0a3a1974476dc3379cea1c430ff73cf",
                        "VirusShare_f0ae091863dc957fa1ca8c868a9c2b41", "VirusShare_f0b57f00186b5f41a6551748e7683dba",
                        "VirusShare_f0bfde0e705f9c38a05635ee21a1e299", "VirusShare_f0e5eb21f24fa68b88f80f3a60a70056",
                        "VirusShare_f0fc440d84ee84b519dd640089d0cafb", "VirusShare_f11a5c4e05c83310d780bf8ae73b3e3c",
                        "VirusShare_f17bd667162974b6123e80d3b06a7aaa", "VirusShare_f19a2d40cf85d786552f220f1adbf0e4",
                        "VirusShare_f1e423f41c86383fe28d697f0e9c3243", "VirusShare_f1e987984c2ac442b81d3dee97f3fc26",
                        "VirusShare_3163055b0686b3ee283203aea56cd940", "VirusShare_319ecadc33caff06e4e8fa4fa9850162",
                        "VirusShare_3334a452a74839af6f19c429b6079dde", "VirusShare_339998460c17d615af9176648f9458a5",
                        "VirusShare_347551439fbcea6990b7796f42b05b15", "VirusShare_354de17131108159ec5f540af5076af3",
                        "VirusShare_3551e43aa991d7a26baac24d7e981f50", "VirusShare_390b6743a93f34c0efa8c4237d58ca15",
                        "VirusShare_3b5527ae5085dd02daf2d4b47247d85e", "VirusShare_3bef8d19b559fb8956e74e17eede8d62",
                        "VirusShare_3ce889259bfa1e97e5ba3549c439c415", "VirusShare_70aad37cbb5464e28e84ab8daa0f0914",
                        "VirusShare_71e13de04cb9f1425778b37c72fe833c", "VirusShare_7462edadd4df98c8096da8197b77f1be",
                        "VirusShare_74ae3c5c71a311dcc7799ffba1efa287", "VirusShare_754ee72d44dc61a41a69411b0528c147",
                        "VirusShare_760140ec6b8dd729d2d41768466af2a4", "VirusShare_760a3d3b0d9404572e940fbead0bca0e",
                        "VirusShare_7638355bb0ea0c033b642dc70ccd2f7f", "VirusShare_784c960d639cb77d17b6702cf623f075",
                        "VirusShare_7912b0ae583f27fdea2cdfa9c46a9957", "VirusShare_79b86733cc8d5b0fcb05e8612cc4ed8d",
                        "VirusShare_7c1740fe83d92dcbb3335fa4f8b26da8", "VirusShare_7c7ac9eaeaeeb73d8903518905930879",
                        "VirusShare_b0fb28f49cc7684fc1f86ea98bcd4b89", "VirusShare_b10052dc459c24292c80f5c655b5decb",
                        "VirusShare_b30c85ecaf27488740ac2e55e5600821", "VirusShare_b315bb152d04c1b9a339080efc3ab780",
                        "VirusShare_b347e07a8173fcbf7a2447d29c79003e", "VirusShare_b3518c38974659e61fefd83725bda7f9",
                        "VirusShare_b3710cf80163dd1066b741ae6cdd5434", "VirusShare_b3a0e330f8b5e1a9a59011dcc282bfe7",
                        "VirusShare_b3cd28799ba9120c736c88cbf3109da9", "VirusShare_b4e1166a4fce1ad54a6184bb2e5aba80",
                        "VirusShare_b4f4369fd47354807f2f83ca54d6f335", "VirusShare_b61968c104125a84a0927c6a2e349ec3",
                        "VirusShare_b6351f03c4af53bccb05f7224f0e9aac", "VirusShare_b6403640daad8737fea5a2b56db82dae",
                        "VirusShare_b6738bb47a07ee08a480bd1620712922", "VirusShare_b67e84d16e47adb24277835b09d2b06d",
                        "VirusShare_b68821624259e7b95a13fdaf247f0b7e", "VirusShare_b6f6706492b34afa1575bf179db97e86",
                        "VirusShare_b6fdecf799700a801a5e9bbeddd23117", "VirusShare_b81aad1e1f1bbaacc43dd5d746bd6795",
                        "VirusShare_b8230f0979607830daa0c4d1f5b5860b", "VirusShare_b8480dd370f2734f73df836087918794",
                        "VirusShare_b88059929bdff8e37d0293804e449133", "VirusShare_b8dd2da46afad4f5c42512ffa9832aad",
                        "VirusShare_ba0b571d23059db41c826adaf3020d3b", "VirusShare_ba298a98b1cb4926d5a70163ab760095",
                        "VirusShare_ba790e0ae6d00aed9223e352d7eef6f1", "VirusShare_ba9f79d4fad43cb91b825c0430c31288",
                        "VirusShare_bad5031f149eafc8f00e6dbe1f178e7c", "VirusShare_bb03e625cede65c203ec5ca267c504a8",
                        "VirusShare_bb41ddf9bd363e80a0526d41be4bf4d8", "VirusShare_bb4a532de186913664d96cc3605cdcc1",
                        "VirusShare_bb6a70c078b3051c310c26e99103d5f1", "VirusShare_bb8b4c3f983b64bae6ae59ce7bd3bce2",
                        "VirusShare_bd3410e53a0b6216bdd3cfab3f92fd67", "VirusShare_bd3bc5f8072990c2a7918aa623914c29",
                        "VirusShare_bd4a7177c357f1d1b462f9d2fbff8cd2", "VirusShare_bd4e1a18bd1fcaa4920b769a5b8d9d27",
                        "VirusShare_bd74b23ad302f5f8985573f7a8d98209", "VirusShare_bd85a2e2805fe1c60de6c0aa4205e739",
                        "VirusShare_bd907733169deb43db9961206f87face", "VirusShare_bd99446b2bdf73ed0593f5d3db008256",
                        "VirusShare_be0c78844ff0987ef2c7556d9587359b", "VirusShare_be15b7d3a816c886ed0f87bac883ec10",
                        "VirusShare_bed38144d7f935072a3d3de04699193a", "VirusShare_bf35631f54bca3b951b11e38dd3e1aab",
                        "VirusShare_e0076bd42ba87f04fd5146e60467642e", "VirusShare_e0222a5cc021b90c1009bc74775b1b78",
                        "VirusShare_e085f4b4ec45f96a1a9bf77659e8053e", "VirusShare_e090887dda06f779439ff199f365baa1",
                        "VirusShare_e096695a0e3891bbdcf77c97cb46e012", "VirusShare_e0a7f3efcdfe831de1a1fccf80d84f5a",
                        "VirusShare_e0b704b95c29a35b6b2bc0c4f366e0d1", "VirusShare_e0d875b40f0f920dfa5221e9032030b6",
                        "VirusShare_e0d8e75a70d7dcb540aae07a448a4880", "VirusShare_e0e294ad1c0c42803af20e4cdb74ddeb",
                        "VirusShare_e12a1f48abbc13236a1bb6059b074005", "VirusShare_e13bf19baac234e8feb4e4924600c316",
                        "VirusShare_e141a12417a9649395f3fe41de2cc389", "VirusShare_e173fa3fe4006dd0de4221ba902f892c",
                        "VirusShare_e1806bc8ec5e0d0284fdbeaa06a99794", "VirusShare_e18ebb486bca007bc3186d0483c049ca",
                        "VirusShare_e1be331837a4f5f1c6a75155bfd4dc73", "VirusShare_e1e289c9d2a4d3d87ea2dfd234076994",
                        "VirusShare_e1ea2d6b06354fd354dbf28acfc4ada7", "VirusShare_e1fe92728f52b0f849c4f11beec91fc5",
                        "VirusShare_e203f1b81f5a118e02fc2d5f123fde2c", "VirusShare_e2314b651c1c999ddcd2e787e9e2344f",
                        "VirusShare_e2945e4cb663b7ec2f821cb77ad7e345", "VirusShare_e2f3258bca27f25bf96621a9b4d8c6f9",
                        "VirusShare_e3245179c974257daeb338381aa6df86", "VirusShare_e36118f9fc60bab39ead623a77032556",
                        "VirusShare_e3983c8775367ff67833a7fe69a7bef4", "VirusShare_e3bfb26ab5d45e5fa1e07128987247c1",
                        "VirusShare_f0778262ffab78e5e938306901abd51c", "VirusShare_f0a3e064b970b357205fc75886d7b562",
                        "VirusShare_f0d9d2ef10aeea681c1bc19b05f1770b", "VirusShare_f0e7e596e84c203f1f0454b59a94ea13",
                        "VirusShare_f0ee0aba90e9003a8d0dffaead3c6721", "VirusShare_f0fd9ccfd5afc9785b1eadcf4fc65f59",
                        "VirusShare_f122508e4f4461f49e6d4f81c9495dae", "VirusShare_3136c24655c11a3cd0c3d9fca0537a26",
                        "VirusShare_31c468d82865fd7b49c26dca84023dd0", "VirusShare_3b49d651cf4ec3411e99af6ed6515760",
                        "VirusShare_3f8798a35d198c72f5c336db825243a0", "VirusShare_701a36ba4f02998c5d974626418919c0",
                        "VirusShare_71b0e59441383082a3abe29280e1aa80", "VirusShare_721284cb20bdc8be17ae0568895fe440",
                        "VirusShare_73b2c62ff46cb800d6169cffd4384df0", "VirusShare_748a6845a04b6a52d088e2aaee44d6d0",
                        "VirusShare_7b28bee3bb5fb4c52ab983b644619720", "VirusShare_b0d9e2427b88492e6c6e7de0bf42a530",
                        "VirusShare_b0f38794c286fecd6c0746a3949469a0", "VirusShare_b127fd74c47f5bb28fff8c93609f7270",
                        "VirusShare_b1453f57e558e2eb4e89a83b13985400", "VirusShare_b1877cdbeb8b566736c5e5f46504d460",
                        "VirusShare_b1cc0f577dd151bf9b854e334acc8bb0", "VirusShare_b30dda594802f06c2c6db2661ddaa970",
                        "VirusShare_b37af46b37db3b30a945aa307d8fd7f0", "VirusShare_b3915dae6cbc49dbf32828bf2da38760",
                        "VirusShare_b3ef729d4aa0409b18ca215841d55060", "VirusShare_b3f666ba7370b207472c8b39fc495cb0",
                        "VirusShare_b3f6728af43207d462b17593ef1c96e0", "VirusShare_b40da6e2e58570791360b11e7e9f1f60",
                        "VirusShare_b4a2304300b12a2c2ecf966961333ff0", "VirusShare_b4bc3f13f2fd7e3857447d1a92752d70",
                        "VirusShare_b4d25b8a8bb2dff6739ba8974342f950", "VirusShare_b4f463a0295f8d5d6c9c60a13a084bd0",
                        "VirusShare_b4fd87dfa07cc46a5d1f6854380b61c0", "VirusShare_b60f124d1593f19effedd4308089ec60",
                        "VirusShare_b660f3e985aaa9fac92b854c33da9510", "VirusShare_b67279bcbece09ce2e6745b21d523ef0",
                        "VirusShare_b6b39b8c6a6dbe47dc4f21579327b580", "VirusShare_b80e0d74cb8f510d4ef1ffe4cc7d7480",
                        "VirusShare_b829a7fb08cc37eea8575d437ee0d1e0", "VirusShare_b8464ecdc342c2177c75c350a86aa960",
                        "VirusShare_b85ae0a7c1d24bb5f0b850c8936dd780", "VirusShare_b860293033d75ae459fddb8120b52720",
                        "VirusShare_b863e07cb6b93b9af342aa3c90992f10", "VirusShare_b8739e9ee22a0bfc576b900cc78ea350",
                        "VirusShare_b8a1b9baa52cc650fd6e8c1c79676d30", "VirusShare_b8a6cf3d7acbbabcacd6a5a26124eaf0",
                        "VirusShare_b8adc138684e49fc28fb051592819d90", "VirusShare_b8b67fe4a5bfa89092f0d8b078194380",
                        "VirusShare_b8c64f1897bb35346f7a3ced5c17b1f0", "VirusShare_ba09a6153a6a95f2257f2fdf82226060",
                        "VirusShare_ba0de5200caacaa953faba2eec203f30", "VirusShare_ba2f229b53a1225cbcc0ef3d8501e070",
                        "VirusShare_ba78fa9f76c76c207b192422c953dd00", "VirusShare_bac2b3ac1424178334e9cf5d17390310",
                        "VirusShare_bade2a5fb6a45f6e8dd1d3ed358f9120", "VirusShare_bade8b3e04760217f6c81a91516fa7c0",
                        "VirusShare_bafde331f6f2207a9484d21a7d18aa80", "VirusShare_bb0c213111ca382f831ead95ee96add0",
                        "VirusShare_bb18923e59d2616a28f8acfb7e40e290", "VirusShare_bb3e78376ef138aa31fef76ecb7f43d0",
                        "VirusShare_bb4423d9455be6e62992a5d78b63a3f0", "VirusShare_bb4d6b58de27c428a650346345ca9a00",
                        "VirusShare_bb69b3b58efb6a2d25e5c8ec1097df30", "VirusShare_bbd8076371b9bb0d88d9f617b0a112e0",
                        "VirusShare_bbefe0dae2876ee3d175ee05356046d0", "VirusShare_bbf209a52404e1a4286bb04ba7f5d4a0",
                        "VirusShare_bd1963fab983359820c45e145daa3cc0", "VirusShare_bd4dc02515f6b8155b3a28bc00a9b380",
                        "VirusShare_bd8eda579fbe8d1e871c11c024b35d30", "VirusShare_bdcf40f12c186d21f6c5b71bae9d7c10",
                        "VirusShare_be0d76ceed3b8af9b19931fc8ede96a0", "VirusShare_be1747150ad3b2728b097fe76410cb20",
                        "VirusShare_be43c443bfb895cf3ddeb6e8f54cb8b0", "VirusShare_be80573c6e4407771946b2b75ca6a230",
                        "VirusShare_bec56b6971d461e5ae4a9ad53ba3b3f0", "VirusShare_becc4829756ae77492e1824efc58c550",
                        "VirusShare_bf03530f79c84dd74b8f2e47785f7450", "VirusShare_bf0e7f03b323ebc8314e4fedddfc61d0",
                        "VirusShare_bf14122fccb8cfd81419c38307177e70", "VirusShare_bf59e0e1938fe7a557a5f81f0a3e93a0",
                        "VirusShare_bf67d30b81d66dbd41ce63501ea6b5c0", "VirusShare_bf745d950c87ab3bea3b55f8bf5d4600",
                        "VirusShare_bf822f61de68ac913afcb50a9dbbf290", "VirusShare_bf998fafa977ce93a94e617e1cd24d40",
                        "VirusShare_bfaadc5f593939a2a126f750cd7db0b0", "VirusShare_e0134e5179fbb70b80f4f55fe2cbc100",
                        "VirusShare_e0362cf5a962b090aed131714e24a810", "VirusShare_e07e2e5966621fca831fcb7050c32b00",
                        "VirusShare_e0c3985e114d7eb3a1b2044ec0c89490", "VirusShare_e165847e4fa2266dd486bdbbd840c840",
                        "VirusShare_e1aaec8b53baaa2d7a621d3695f35a20", "VirusShare_e1f91cf7ede1476519b4edd270be5280",
                        "VirusShare_e201e0220b4a572da9f3306ff5d7d0d0", "VirusShare_e259bfa8edb2642c1ed99292fc41ba10",
                        "VirusShare_e2786da65b182a9cf84ae65169be6e30", "VirusShare_e2921ba17e842d064aa4b05d250fafd0",
                        "VirusShare_e2b8b928b072c57af892d8cc7e476510", "VirusShare_e3b3bf94795c75328df6ada1814240b0",
                        "VirusShare_e3f0ba3838a3408790e777db44214540", "VirusShare_e3f1619a0b5386342a9523dd391bc8f0",
                        "VirusShare_e4051d7d65d66c1ead2d7dd28ca35010", "VirusShare_e41403d24740f808a6ca049d655ed190",
                        "VirusShare_e42b7fb6451e51fea584887a020e8050", "VirusShare_e42fa322fceb380200352849fdb961b0",
                        "VirusShare_e436aa82c6100f2f089595bce84ce680", "VirusShare_e445d567199815cea4ac1f28705b2f60",
                        "VirusShare_f00bac42a1a9add8d907a41bf573f9d0", "VirusShare_f07943600c48a067ce2f6377684064e0",
                        "VirusShare_f0816f83eed95c101a1c4b54d1536430", "VirusShare_f13f9f7249a8782c38637860adb43d10",
                        "VirusShare_f172ffd12006a8281dcca532a3d76460", "VirusShare_3204987fadb4165a83c7dff54b5f03c0",
                        "VirusShare_325d41a17cf8c766b89bc71d9e7c3067", "VirusShare_35055496893b05290745b1b5b144c6ec",
                        "VirusShare_353354e5354dd2ca4e3e2a57ceee44f0", "VirusShare_368b69f49dbc2f098a69695d6778b357",
                        "VirusShare_377c52d1caf33f0ba108d2e7f145d746", "VirusShare_37a2ff981af1525d3207308ad4449600",
                        "VirusShare_38364115b040910d6f1b07d0f8043fe0", "VirusShare_39d888a08ed11dffbc7fe0dc7bcb5eca",
                        "VirusShare_3a3a35892813cfc07e7c092786d9ac70", "VirusShare_3aa64e93974175e50c96ed1a29e33990",
                        "VirusShare_3b39ab6628896b0340e286d3ffa56b5c", "VirusShare_3bcc0559abb3b31d9c2707272414ca00",
                        "VirusShare_75f98e48ecc0506f93a0079869fe28a6", "VirusShare_76117adbccb772624824bcf44090a560",
                        "VirusShare_77784c80ae585ac292f39fb254d3d500", "VirusShare_784ac8fce934407173567cc77e751a25",
                        "VirusShare_7bb1675bf1f4042c4114a55229f40e30", "VirusShare_7cc7fb496e8d9cb72f8bfc3436020200",
                        "VirusShare_b0f0c5fa61a2a2b506c878fd1b336540", "VirusShare_b0fd2f67676c81dc85f6e73b7904d955",
                        "VirusShare_b145a65232bcf6c170ce3cdee3448250", "VirusShare_b1945183cc1f1cd3b5ae307671460220",
                        "VirusShare_b1ee70d432c4a4d4895cc7df18dc9730", "VirusShare_b3296d4070b6bf6258412f68a5bfd614",
                        "VirusShare_b3496e1e281350590061b2b4c95176e0", "VirusShare_b35b112d359d974f7e8c8db2562a22f0",
                        "VirusShare_b3ea7b4497bc0559fc994c55eaa1c3d0", "VirusShare_b40fb1ae4993b9844b4e798a819d2420",
                        "VirusShare_b431a826e83a1c39bf0a0c920c2fc504", "VirusShare_b48afe05f530cfb6fdf192f19507a220",
                        "VirusShare_b6076a4144f0eb8a96f241525eab31c0", "VirusShare_b6698129f9648a4494e08fd8bd929db0",
                        "VirusShare_b6741761f4781a7cd9f4993cdf0163e0", "VirusShare_b6f67e16d817a373e0c9275d8ce5a380",
                        "VirusShare_b800970262963a442ea21470d00f9020", "VirusShare_b80f92f75dcd6788932ad7cff9631340",
                        "VirusShare_b8837fa373f340532e7abb7499e0c7b0", "VirusShare_b89aa132126ac6d2db4097c8cdfa9fbe",
                        "VirusShare_b8b54e14dd95ff30b1d130e8169fe791", "VirusShare_b8c90f550dd74f770ad015c5620214ef",
                        "VirusShare_ba0f1ddb1065c30a4664fb2f53711030", "VirusShare_ba154aa8ceb9a6fe3927aa29e1fb9cae",
                        "VirusShare_ba5f00bc51b8d69e8283138be20f257f", "VirusShare_ba6d45a1f08c5d4d26ed8cd64afb0fa0",
                        "VirusShare_ba8e34715c6141439d4e14b6051cac5b", "VirusShare_ba9ca2a618aa18478461653b9c4040b0",
                        "VirusShare_baa4b19c9bf65ebf901abeb6370a9550", "VirusShare_baa9af63073087c840d5d49eb4ca7800",
                        "VirusShare_bac0aeb5fae4d0807b4e1262b88faf40", "VirusShare_bae1847462d3794e40117398fce98440",
                        "VirusShare_bbc850c9c15a947b221aafdbd9bb43e2", "VirusShare_bbe58f482431b8ddaaab5e3fc2d4a72c",
                        "VirusShare_bbf70caae58cf211ffc050aa6e29388c", "VirusShare_bbf9e4cd06e617c9ce19b75ddcc480d6",
                        "VirusShare_bbfc76e29534606cdb5bba426857d430", "VirusShare_bd0281226ac591a5b4444276a077d800",
                        "VirusShare_bd67db2a35375380109bb3e83f87ad80", "VirusShare_bdb4a290e735b272da27573e4927015a",
                        "VirusShare_bdea612220d76b6e2a4c2abec56be923", "VirusShare_be0b593b6b3fd5ddd71b5b4214640b95",
                        "VirusShare_be1ac4eda512d2e98ccb611a46554620", "VirusShare_be2417485dde67bb8df2d300c117d900",
                        "VirusShare_be3e57f140bfc9f5520e7e0b27a01370", "VirusShare_be64218c276b51f73a9e84ca50d19f04",
                        "VirusShare_bea677f618f90934d0ea104cd2595690", "VirusShare_bea904dc61344f6240314b67b7551db0",
                        "VirusShare_bee4076910655c1c923d2edd332d9420", "VirusShare_bf0858731826c63d6fac40ed325f3020",
                        "VirusShare_bf08ba56f48e2f1cdf6b3181bb7a10b0", "VirusShare_bf56222d151053a0c281c5c57e93431b",
                        "VirusShare_bfc25bae96f6f9a9d1865a2941d9322f", "VirusShare_e02e6e7d5a8952b6c2430c56603a16f0",
                        "VirusShare_e0331ff912bbe5e9b8810d77c6f8aacf", "VirusShare_e03f4370531db0b18e103416d797708b",
                        "VirusShare_e0608785ef813f4da96b92803b9c2bd0", "VirusShare_e06686779c4d25a0d300875edef3e570",
                        "VirusShare_e06ea0632ba02c8b27bf458ed2e88a1a", "VirusShare_e0dbd59c928b90248393dcf00cc74950",
                        "VirusShare_e0e25518bd1e17a97eb8342daaed1270", "VirusShare_e109f681fc3acc022a52e69816d35100",
                        "VirusShare_e12f96c0b92712b28a7f5c2aab23bc21", "VirusShare_e13667b2fe64c8cbf5321dfa469bb8e0",
                        "VirusShare_e1409c3d4e417955401c35e6c2c35bf0", "VirusShare_e15727430eedae86170c5a866c42b170",
                        "VirusShare_e16299e2abf010df2cdfdee74a3ea565", "VirusShare_e1ca072453a34f5de4eff64df6ac146a",
                        "VirusShare_e1d24e00cb8aa4e9fb1df83f9397e970", "VirusShare_e1e326ad542656c81e687163ec239e40",
                        "VirusShare_e1f2b15ec9f9a282065c931ec32a44b0", "VirusShare_e1fa8cf48c2f8dc3c6f51b0f1a8a253b",
                        "VirusShare_e22b4323fb90339e12750ba1cb438cb5", "VirusShare_e29952b72fc7577b51191eccf64cf770",
                        "VirusShare_e2ffa82f1f1ad28a181c1f0e98cc8b8f", "VirusShare_e328070b22a04e1b5959351c8f4fa0d5",
                        "VirusShare_e32e8c6c72cbaefc115c0d3bd6525a4f", "VirusShare_e3910ca5e5ae36f410c60226ef677d9c",
                        "VirusShare_e41477fab2c928c62682eed9cf79f5d0", "VirusShare_f0055101d7eb5f6416182efed04eb240",
                        "VirusShare_f02d0a127314d8a1e7c7cdd46065a9c5", "VirusShare_f065e8a2bdc1dabaaccade0094442a90",
                        "VirusShare_f0b3954c7f5c2fa80fe606fd4a57f2f0", "VirusShare_f0fa63c63afcf5b2859170adbc6d18e0",
                        "VirusShare_f11a9ad0b20af7bc4d8181adbf3aaa72", "VirusShare_f133239e8615219febaeb613c6c390de",
                        "VirusShare_f168be14f7abee5d779767f02991a533", "VirusShare_f180c2af847a0fe49eb887b3b1575200",
                        "VirusShare_f1ad3e2aff64a5d6a50b7f864f614b50", "VirusShare_30ae5e440c456140c568601922a2d6a0",
                        "VirusShare_38440b293cf24374fc10351f1d15b852", "VirusShare_39e4ef94e8422333a0360d39ae73d100",
                        "VirusShare_3ac3a4416c077348f3e936319d9aa0f2", "VirusShare_3d98b57d96e5e6c65a845bc104076e77",
                        "VirusShare_3f907e963b75151233dd3bc33877bc92", "VirusShare_70051397254acc40cc66cc78a1f9a4ef",
                        "VirusShare_704b5d5934b388277d0aa3f934630470", "VirusShare_70997cc94bc406243e0878fefa234b00",
                        "VirusShare_7179899b09659e670584897203159ecf", "VirusShare_726674cfd98aac65f3353dddf3d10134",
                        "VirusShare_72f64fa3aa5205b4ea85d638fb862114", "VirusShare_7399ee65297bce1daa7e161a437aadc9",
                        "VirusShare_7431bb2b9b1dc80330eeffe3415185bb", "VirusShare_74b09c64f80c76f561b2d0a3f969270e",
                        "VirusShare_752356f630b27c3c956d1b8e3f0d9550", "VirusShare_758e0ff6019bf1d138595b0c27f30c52",
                        "VirusShare_77b0d100859188eab47d599bfe005098", "VirusShare_7812570b9eb9d9917df6d457160d8556",
                        "VirusShare_79ccd7da36f7e795f9720c087a2bf7b0", "VirusShare_7ce0f2cbbce6c6005cf5239d1169d870",
                        "VirusShare_7cf29f02ce782bf794ccf021afaa2220", "VirusShare_7d42276d6bfe794437fa8541bf7b90cd",
                        "VirusShare_b0de6534d434d8384d49e57f84269920", "VirusShare_b0e670e384808334232fe08c1d1c930e",
                        "VirusShare_b12834e4b67982b76d9561ed682d2b0a", "VirusShare_b1887ac515bde813ded05a14f931e206",
                        "VirusShare_b18d98149b79be048fccdd8cddfe1993", "VirusShare_b1cab19820cdb83f789067caaa52e870",
                        "VirusShare_b3115bfc493e60cf6ee7a3d88eb14ebf", "VirusShare_b31834ad08da7264fc312310331326a1",
                        "VirusShare_b34a1bb9c0cd10104623a9116134579d", "VirusShare_b376de2a165589966cec39b5be5be362",
                        "VirusShare_b3c08a04a67b78fce91c9d4f0eb784b0", "VirusShare_b451d87e4f36b565e9a6e04c5a8fe0e3",
                        "VirusShare_b468a62a12e2bfd6f9076eb7ebee79fd", "VirusShare_b4c8428e68edb90d0b8e4c3b3e76a05d",
                        "VirusShare_b4cf719af0315b20111ec803e3367e8f", "VirusShare_b600bd751ea805039a7588d9614c7d1e",
                        "VirusShare_b60d33de411b734aff09c3811e0ed3d0", "VirusShare_b6548f29658782e40fc69db8cf6a7413",
                        "VirusShare_b6b85dbe8731baf3903a022422d1547e", "VirusShare_b6cc0edb542160157b767fb1f48b26f0",
                        "VirusShare_b859a287257f9836e7bd03a18a7e6fd3", "VirusShare_b87b41b26992f6b93973d1ba34f3a009",
                        "VirusShare_b8869e4901b68864b9617a8bd820c800", "VirusShare_b8aed7240d6face133491496f9a19039",
                        "VirusShare_b8ec8217087da151cf5523f0b4140e19", "VirusShare_ba285d3b07c055e7a5c376cb7f1216cd",
                        "VirusShare_ba59b7cbc9c6887399194f3544fb8fa9", "VirusShare_ba5c001da25cc86634fcb283cb77f496",
                        "VirusShare_ba873b3cdf2990d2981619db9f97c9d2", "VirusShare_ba8b326f1dfe47d5f3a9806c99e9c93a",
                        "VirusShare_ba8e7c59c26f8e27eb4c3413b1333a16", "VirusShare_ba92d67ca98da1a739cb237b68fc0d8f",
                        "VirusShare_bb02f199c5fc620f4936c885887bb9c0", "VirusShare_bb5fe49569940835656a249b2e43e073",
                        "VirusShare_bb8ded6bea693c6f51857afcb41c56a0", "VirusShare_bb9f2c50e221ca5b7e7c445e67d8b960",
                        "VirusShare_bbb009be873a55866f2e75e0dafe175e", "VirusShare_bbb1069eb1084916193c03b1920db140",
                        "VirusShare_bbcfeed82ea3a2d56a87278c53dd1724", "VirusShare_bd22f27a780ca6659c85e93ef7acb809",
                        "VirusShare_bd9ada83bb75da6fff6728b19530dfbb", "VirusShare_be2b719a2f71c1326fcae57a4db2847b",
                        "VirusShare_be98225a6d984edfdd177c589546fab0", "VirusShare_be9918c7cde3a7c803a5e648b37dc035",
                        "VirusShare_bf1271bb093b04abed23a3517273305d", "VirusShare_e0196d14964cb410a4453232db56572d",
                        "VirusShare_e06ec615d1b6642fb69c59f9de191f3e", "VirusShare_e089d19ff1f221b9cfd1287ff4a0cdec",
                        "VirusShare_e0a839d8b9e08f1275c1528343a893c0", "VirusShare_e0ba77bedc7035bc7c4a4f008606806e",
                        "VirusShare_e0daae8b1cd59f9e4dff3daa89b6f7ae", "VirusShare_e0e4942379dce9891623ef868b408a5a",
                        "VirusShare_e12f3f4a72ec3142a77dd0869c6075b0", "VirusShare_e13583f100f500b8caec7e102aa9b22c",
                        "VirusShare_e13a600ebbb6000eb8e3d3ce3f95cd53", "VirusShare_e16c408e0e83e98926d9ea79625e01d6",
                        "VirusShare_e195f86091a6340ab3fa60d156d0b240", "VirusShare_e1b2fd9692b44ae6d95c768370273de0",
                        "VirusShare_e1cd755867ff9f04dadc55f3b2d2630b", "VirusShare_e1e5b262fc391604a8487c5b42a43d3b",
                        "VirusShare_e1ea42286c8df4f093bff8636daaf38b", "VirusShare_e2c151f5fc844df69a5f536034d694bd",
                        "VirusShare_e2ff3b1f992ba3d8c294d806f8c17938", "VirusShare_e32910526a25e11a78ec781a6ae56c34",
                        "VirusShare_e347523831517cb4e3777bc72c930080", "VirusShare_e36a2cf01fa7c0f6e6f6dc0821bd7b68",
                        "VirusShare_e3770901c285a8a4bee6f37ba492eee9", "VirusShare_e385f50d9a53d28d54e266bc843109ae",
                        "VirusShare_e39d3b47f93a434bb011ba4e7210cde9", "VirusShare_e3ce1ac6bd97e51efea50127ed85191f",
                        "VirusShare_e3dd978c63ae43b2c0a4db6eb74a66d8", "VirusShare_e3deb708e02d0eff1347c08785accd34",
                        "VirusShare_e3f7f3a8139e182e42145df3cdb5abc8", "VirusShare_e41544cd49894241808f43d5b988cedd",
                        "VirusShare_e41c3451a760c394493602d37f80ebee", "VirusShare_e42690e4a5d5e04940c6079f40adbb31",
                        "VirusShare_f00d6a23915743d6b44268f108f64e4e", "VirusShare_f02984ce5e52504fd5e703a02a8c48d0",
                        "VirusShare_f05661afa04b92dbc17a7102ce774966", "VirusShare_f063ed5822b53adf180397efbfa51a80",
                        "VirusShare_f09593d2a3aeec33bab430c65a2a61bd", "VirusShare_f0b59bfe1f975aac30f939f5905196cb",
                        "VirusShare_f0c6de286b0a1e9c66b157f8a76f1583", "VirusShare_f0ee40ae10194541039f5404fcda3f4b",
                        "VirusShare_f0f3f1db89fcb9719173aabcc6f5bd60", "VirusShare_f168f608bf61899304b501752b2cd380",
                        "VirusShare_f1a9c625427ae0f05c9b7570a557bb01", "VirusShare_f1ef14794563b580fce0f3af32afc491",
                        "VirusShare_3413ae6c501845858e3cf16b10626660", "VirusShare_34b3ebf588b1cd5dcf8370187308ce2b",
                        "VirusShare_34b483d21ed7ef791d1cf64af309614e", "VirusShare_36409f02114e0e13460375f1ca338a1b",
                        "VirusShare_371902f578527ee65e0d7a9e2714b30f", "VirusShare_375046b15ad85b75a2fbb3e9d07e86e0",
                        "VirusShare_38580783fe834b192c19b81f1c59ba28", "VirusShare_3d79626c45197bd14b1ac2d5bfe63cd2",
                        "VirusShare_3e820f92cf249035843cec73ebd13596", "VirusShare_3edc8a599442f48d1892e6b12494da02",
                        "VirusShare_3f245a4309d87c38ff6fcf4df424f08f", "VirusShare_3f57d0055adac588c3b3dfdff4ef8ee2",
                        "VirusShare_70dec1568e0ed5753dab8df82170cdb4", "VirusShare_717ca92d8fc11723416b0b63bcdc5e13",
                        "VirusShare_7197348b46b63bcb95f5d895c0af885e", "VirusShare_7312f7bfef00edd5325c54e618fa5f48",
                        "VirusShare_7335b22cd00a6248b2e12c966f9a93dc", "VirusShare_7387d391adeaba71e8b10f5a1aece28c",
                        "VirusShare_73b100ced5b728f0a09d7d606377c63b", "VirusShare_74f827a931bc88f0e3177ad4e89d082a",
                        "VirusShare_7786084894b10f0823b35b06b86b9972", "VirusShare_78992cfabd6b8cfc2a5aba88c601aad2",
                        "VirusShare_793d7d48ff2ecdfa3c97bce3b1bfcbb6", "VirusShare_7a8d03742d3578ac23d631dc15e9672f",
                        "VirusShare_7d2448e015e34a1a833d6ead1d86a39e", "VirusShare_7d453d61125a4146061d879695579c77",
                        "VirusShare_7d4cd3ab681a5ee9314a7c97a6bd58c9", "VirusShare_b1496c761e354a190cd46279a4e17da0",
                        "VirusShare_b14b5c10385a288cf646613b5c08109c", "VirusShare_b192059fa8801382f8c4d94cd86cf6ce",
                        "VirusShare_b30a8db651b8f217d4d96ba73ee0ee1d", "VirusShare_b32da31f6bee832135e3ea2cfc26221b",
                        "VirusShare_b35206de8fb12db36742a1a9ce6dee91", "VirusShare_b36ee9c7e68a486964eada88148c6d30",
                        "VirusShare_b37a1d1d55a3057d345f6678ab041ab2", "VirusShare_b3ae2a11dfea56d1a5e52797b941bfef",
                        "VirusShare_b3b956ae84e70c9d06c54a2558774945", "VirusShare_b46926a9b28c2e0769b88efa97507c70",
                        "VirusShare_b478c6fd409ff761ecddaba39799f6e1", "VirusShare_b48c8ee96b02cd3715046a9610f810d0",
                        "VirusShare_b4cf569ade6cc1ae7a6797f71df7855c", "VirusShare_b60b09d1cf07a77646bd71799f1198e4",
                        "VirusShare_b6160903e04c9bd26adf9d0582aab586", "VirusShare_b6a300ddeb2fcd9e1efa34cc57596ff8",
                        "VirusShare_b83a9d0fd01de1cfd3ff1f8ddcadf6f9", "VirusShare_b869c1b46b9c834323b7bbc9e92b86fa",
                        "VirusShare_b8967ef4e2b770296a95dcb227498fce", "VirusShare_b8a53413a3e41d50ed9250d812512570",
                        "VirusShare_b8aa27f08868cf5bead5bafdd972b68b", "VirusShare_b8da83a0354b6134a4cf8250ff66cc5c",
                        "VirusShare_b8f9b63664785cf8a498873c8fe007c4", "VirusShare_b8fecf2279ec973bb544b2dc6d32d44e",
                        "VirusShare_ba1d7462bc1169f7e8353d126cee08ce", "VirusShare_ba21d0f7515645420464a4ec928e0920",
                        "VirusShare_ba493bff7aee0111f5b31d6a610dfaa2", "VirusShare_ba70dc27df52f25f2a0201cae765430c",
                        "VirusShare_babac91f023eedfe6648504efc48f380", "VirusShare_bafa64175dd54e936b761461ad336f72",
                        "VirusShare_bb4535558b41fb24ee99372e4c9410dd", "VirusShare_bb5ce53612e530c63f2e2fe8bd15774f",
                        "VirusShare_bb5e008a8ecbe8c4a486330aab730e9f", "VirusShare_bb91cab3b07215851a39c581e98d9fd6",
                        "VirusShare_bbe6dd07e742a175d3f4e4e9619f8c05", "VirusShare_bd8cc96b021835078dd0f13123a347ba",
                        "VirusShare_bd9f5877b6a8ff49d0254d2bff3a3ade", "VirusShare_bda9c64dc2a72b25b47f23503700cc8d",
                        "VirusShare_bdd00f8910e2604b8258670ee1d4c7bf", "VirusShare_bdf107137147c73c7414b5946c54b950",
                        "VirusShare_be25394e32290fe9b82d6f2921a42243", "VirusShare_be43beb2592f46b6250f7f2782bdf175",
                        "VirusShare_be71fcc5ab23c81e80102bbbcc0605cd", "VirusShare_be88c29c68fdfbb5ee78795b2b6f14b3",
                        "VirusShare_befe0645ee4d842dcd6b631bbacfe1f7", "VirusShare_bf7f6aac6b53eda4b40f0d79cb1403cd",
                        "VirusShare_bf8236a3f5b9bedfeae16ecc0ee5702e", "VirusShare_bf8885225e5faf605b5f11911e2601c7",
                        "VirusShare_bf95722424dd01a11103664c5af97e78", "VirusShare_bfd154cea75126c3aeb5b1a3390b1768",
                        "VirusShare_bfee6fdae042efe79ec4dc896a3b8490", "VirusShare_e021f100281cb65bad63a488a85d4720",
                        "VirusShare_e032b43ff911d80d29220d9e61ce332e", "VirusShare_e0450d39d1694ae9951e42de1b4182b4",
                        "VirusShare_e04b693eeb276060ac88577f241eac19", "VirusShare_e066c145fb284106effeff06a3b849ba",
                        "VirusShare_e085e0b1405195e77029072270e0d633", "VirusShare_e092fc78eb81c1a8af1264436198c5cb",
                        "VirusShare_e09f7f8b67df74e5c57e11d8bf257e90", "VirusShare_e0a4870b765e01e7fcca11bda02f9c31",
                        "VirusShare_e1249438aacbb194b7efaf7b18ca6f00", "VirusShare_e14e4ffb3d75e110520ca7e0141fc843",
                        "VirusShare_e17648eb25db88388ad4ad3fb3d0bc5e", "VirusShare_e1a0b4b3f7ac2a343b0cc438d8d224bb",
                        "VirusShare_e1ad9bd68bff7f728af3b94a7e3010c0", "VirusShare_e1e9c98ce73aa749549deb42677a9a1f",
                        "VirusShare_e23c05934b40f44e80cdd57e8b834b8d", "VirusShare_e257a0c371f78149b216a7d116cd54cb",
                        "VirusShare_e2c1f0b8e8a5c3099b2cb95be7b19761", "VirusShare_e2cd5ada0d16f82327aaa52344f5c954",
                        "VirusShare_e354a7f025d0fc493f02540dba74e7ed", "VirusShare_e39d614832602bd186c5b95e3898e70b",
                        "VirusShare_e3b49b9f1dba3b8f54701e828a65f1db", "VirusShare_e3d3fb68ad087225c3811ce1a02dd2c0",
                        "VirusShare_e4095def65675a719e1152ae920765eb", "VirusShare_e4488cd9b0456d5dd0f3daa7daebeb0c",
                        "VirusShare_f00e56e10e3ca8b9a76da95be067ec74", "VirusShare_f0508a38db1fbc8171a74ee2e2cf196f",
                        "VirusShare_f0ffa82e164544dd38e41e3a37a72333", "VirusShare_f1161e13b3386acacba686983c21213c",
                        "VirusShare_f15ff94443ba19ac407ed21c2025fbff", "VirusShare_f16921748a5cbf545285e4d2d5127459",
                        "VirusShare_f1711635350a067be87b2cf56b0ed8bb", "VirusShare_f1922c4522f8dd1d8865fab853b60daa",
                        "VirusShare_f1f798894df3285e459e162b99b1706a", "VirusShare_31fa48eb094209580f416bc11e61f130",
                        "VirusShare_32b19aaa7dc6f651d6e2f8d9c5aa4240", "VirusShare_32dd479e4d09afcceb709c4d674cc910",
                        "VirusShare_33fc0f41a3aa00f76613c6ff4b6f4c30", "VirusShare_371aa8d26c8243e6e77cf59ff389dfd5",
                        "VirusShare_3c5e5f291a270f4742826bd7300f84f0", "VirusShare_3d6b76852262c259f0945e9964c83be4",
                        "VirusShare_3e08bf041e4f3d4b43277a0db634d320", "VirusShare_3f0c764bbf3497714e7c26ffbef337f9",
                        "VirusShare_3f795ec8f5b64ffc280dc5c430ac07f4", "VirusShare_705357ad53ef835029c6b27f8ad0ccd0",
                        "VirusShare_714d84d13b8e511cf002187029194cc5", "VirusShare_718f53b1620d70f810c9ab0a2737a23a",
                        "VirusShare_7211c5a485c18c906e98163fe9b72790", "VirusShare_72807929c8061e2283a3377f5b3f8fb0",
                        "VirusShare_72bbc69e7f6499f1ed94a1bff2aa1a20", "VirusShare_74e1e9ae37c636f79371bcde6c6c0440",
                        "VirusShare_75a500f3f7c0fece768e5609d7584c60", "VirusShare_75d7f76047e81aaeb2da5e426b974ad0",
                        "VirusShare_75e97a2d5d6955f31443befdbfb25f30", "VirusShare_76309b93e028fe7b2c1c2e04f11c26e6",
                        "VirusShare_76955d6aaff714149dcfb5c95b6257b0", "VirusShare_77c2e688d93f0eb51e28bfa96026d023",
                        "VirusShare_77f83af846fda38f2bd1873a2972b010", "VirusShare_7979f867d5805ccd6a92d05eaac71030",
                        "VirusShare_7a123b059a0e45f93574b68cf3ebb650", "VirusShare_7a86dbd468bc50d80bee3931a9935ad0",
                        "VirusShare_b12662d586b31f0e8650f0ac95baec60", "VirusShare_b12dfc3f6612c4168a2ddcfdc12d2a70",
                        "VirusShare_b181cf7210163cf4d33265cd2506d130", "VirusShare_b18dcf59f004700e135b9088eba82f90",
                        "VirusShare_b1b4b9aa6d51ae76f0fb153a6dcfa350", "VirusShare_b31f32685b41a708f8b5f0ae2918fe40",
                        "VirusShare_b339dbb0fec59dca4ac35c0661f23610", "VirusShare_b37481ad6c2c7c0fc5e04d74fef54150",
                        "VirusShare_b37ab8a94e29da446496fb5252369b0f", "VirusShare_b3930b0e5b6d6d36d150721ba4a8ab60",
                        "VirusShare_b40a6d5475d6ca014d18f5148d4a21f0", "VirusShare_b441fc571716279dedcb6c2e217a1ea0",
                        "VirusShare_b4931f0dd5ba37fc9e330f05ace3d5fc", "VirusShare_b49cb15bb2a794fd7028924d07973de0",
                        "VirusShare_b4a710bf7edb48caf32209fa48569e10", "VirusShare_b4acfa505d1135fbb4b13f1a82970360",
                        "VirusShare_b4f5b67d37b3d28fd5e97345edf64da0", "VirusShare_b6bcda8eb7b90682190f518169bf78e0",
                        "VirusShare_b6c1a9f352756cc07f866e50a8b61140", "VirusShare_b6c2068dccb76232f68883391613ff40",
                        "VirusShare_b8467c2f83dee555d56b9634d97eb340", "VirusShare_b84bc64d6f02c0e2c0712f77ec5f0b3a",
                        "VirusShare_b8695f80827f70dc77d10f3613e196cc", "VirusShare_b88906cee8517509204cd5fb811751e0",
                        "VirusShare_b8c1e1f242fd87ebda5f7f4c651cfdc1", "VirusShare_b8ce29144865d7572b5ed963cb0b0af0",
                        "VirusShare_ba3a2d4ea9607c2a0d7083e9d06336b0", "VirusShare_ba3cf96d84fa3ff67236b9f0cc146910",
                        "VirusShare_ba47a5bfea5e76f78064ebec6524d3d0", "VirusShare_ba4a2d2832bf822f3aec4bb55a3010b0",
                        "VirusShare_ba5d5626954137f25cf0fe9bd8ad0580", "VirusShare_bab490c29524406e8ada9c518af7f280",
                        "VirusShare_bb2f0ecb241e9f0e4a81c197f7ce9080", "VirusShare_bb43bf40ea2da8eb20afa41a8371d190",
                        "VirusShare_bb9b9f9dee6ef3dd9bda74bd16387510", "VirusShare_bba438fbb2ae6e0042ebb17da3a7f616",
                        "VirusShare_bbfdb9a4a4c0d33ea9c563b9f887f3d0", "VirusShare_bdb7586a6199ddd43e75875536c1a460",
                        "VirusShare_be053c0700d805c92d3e0a1c1418ba90", "VirusShare_be1bb124e437359d5aeadbd09b4c2c50",
                        "VirusShare_bec4bfcd4b5931aa0dfe270c3cbfa1e5", "VirusShare_bec9ee5e4bcce412b01d08c3245b5d20",
                        "VirusShare_bee3e9ace039db7d3283a6949b6b2520", "VirusShare_bf03a57f7cb39ac78e0a798e9a29e190",
                        "VirusShare_bf152eb430c3bb2a147b84276a5554f0", "VirusShare_bfa9d481085e05f58875f405a597f4ae",
                        "VirusShare_bfcc7f064467564d9076daadb5d2eac0", "VirusShare_bff23724b9b3b144ebf32b30064803d0",
                        "VirusShare_e01b0f05460b6852fdcdfafd3d97cca0", "VirusShare_e01d74d37f19f2af2ce8d9f01f0df300",
                        "VirusShare_e0413c36fcb2c81396e61e259db21a40", "VirusShare_e07106d53f7f73879b9d397e6e5eef50",
                        "VirusShare_e0a6b464bffb12879eeb20fafdc41563", "VirusShare_e0fd1930c2ba4bbac7994c6994a66d50",
                        "VirusShare_e11dc52f27d7979fe01ca6ff4f005cf0", "VirusShare_e152e11115ad87a806648fdf30acd230",
                        "VirusShare_e1a9d721bbd8b2a8f2d8b7ff20d39498", "VirusShare_e1c0dc001224f9a1484070a2e2416b00",
                        "VirusShare_e1d9ca31b0d250a59d696ac057f45200", "VirusShare_e1f874ff8841339ccedf506394546b87",
                        "VirusShare_e227b412c4b286adf5c9cb00ef95cf89", "VirusShare_e24085bbfd24209de94f5744913aeb60",
                        "VirusShare_e274bcc9cf4d245e321ac38672584040", "VirusShare_e2935879321ca470ca5e0e3936e01090",
                        "VirusShare_e2d3aa2d5eec15b74be16d67d1a07aa6", "VirusShare_e30899478b0b2fd46393a04fea9d0130",
                        "VirusShare_e3250ccf731cdfa58cbf62924d05bad0", "VirusShare_e3716518b55cb6ac62f778094e8d2540",
                        "VirusShare_e39c7ba41e80b0e576d754413854f140", "VirusShare_e3acf8c1a0412e43fff16a1729ca160b",
                        "VirusShare_e3d5557067df231de44f1837cd8f50f0", "VirusShare_e3eed025e57e14f2c752cb6cd7210eb8",
                        "VirusShare_e40e5c2ded252a7f284e5c63eaa1ee00", "VirusShare_e4365ba88e6ba913fb458ac1f0a59720",
                        "VirusShare_e44cdd7b9d3d57e0a2bb97710c9b0d10", "VirusShare_f002f62d05b2f845c90bf99159c2a734",
                        "VirusShare_f01b69462de7cf79ca1317bca167edf0", "VirusShare_f088e716e51abe4a29d10ba184a69083",
                        "VirusShare_f0d572fdc498c9dc9ae4a49669407290", "VirusShare_f0e214f2b4d747a531244acfe76ad3f0",
                        "VirusShare_f0ea73757540b73575c9be49f9890300", "VirusShare_f0ee189f564352e615ab63713e12bfa0",
                        "VirusShare_f0ee51ee19050f12345120b716d6a0d0", "VirusShare_f167f8150b3971b5557a236acd7a3e6a",
                        "VirusShare_f1b87be68c2195c10d0fbee23b118d12", "VirusShare_f1cfd39c70c050f04de1ecf807b5b240",
                        "VirusShare_f1db9d003a9253b9d9682e5fcef07030", "VirusShare_3038e98f9a221ee544d3a6f47315e4d0",
                        "VirusShare_33de473a2e0470fb0c161d10c607c97f", "VirusShare_347ba6c63ed4eff78f52def45ea19251",
                        "VirusShare_3576df09d4797ce04ab421c69f335821", "VirusShare_365a3c0fc8fcb0a43112eb939ffff0c0",
                        "VirusShare_37880eba1ff494374a861f8d7f5ca3ed", "VirusShare_381a0c792d5c1f1ed111794b8c565706",
                        "VirusShare_386540cc57b46dcb2bd9dddfb1fcbd40", "VirusShare_38721636b5bb09b1622f1b49cdca5084",
                        "VirusShare_39468aaf4b2779b9d92e6108375b0320", "VirusShare_3c1710225d006425dd4c563eb53ec5f1",
                        "VirusShare_3cbd88b7ef7aff9e3f9b367807c239e0", "VirusShare_3ce5138375911d45bcc409f554d49158",
                        "VirusShare_3d8d797c32243158d295b873d7a6bc60", "VirusShare_3e2ff630dffb30f70604f89ef71c12b3",
                        "VirusShare_3ea2b546ff4ee9bcbfacb9695f9e06b0", "VirusShare_3f56d4406a8df07c6ec57dbe0ecb9a10",
                        "VirusShare_3fb3bee414e86e63c39ba0921fa2bc64", "VirusShare_7182306d3c49856d1d8db6f525e8de10",
                        "VirusShare_72764cfb48a9d427d845500066cb5364", "VirusShare_74bff00b068dcb91825920368bdaaf36",
                        "VirusShare_74cbcdcb6979189bef1a3032856b7550", "VirusShare_77de9c6167f086814541290c82ac8600",
                        "VirusShare_78047f8243aa48fee5b59f9d0654a9e4", "VirusShare_7a39e3551a857ee418cf2913d25330ca",
                        "VirusShare_b342506a2873f9ce3b4123a3f8c7da50", "VirusShare_b37806b13c0fab80809daf71ba3241d1",
                        "VirusShare_b3842e4774acd9d9993b708016cc2baa", "VirusShare_b3af542a394d7f4cc6c2cc5ca5f569f0",
                        "VirusShare_b3b3ec3d76931366f673311eda8f1510", "VirusShare_b3cb3c616815d50924f5265cda006430",
                        "VirusShare_b3f27443274a37c30928ceee536ec9b0", "VirusShare_b3f53ad7c93798d01a22e3fd7ec3baa0",
                        "VirusShare_b3fd666ca6df50e2812e9e8491dc579d", "VirusShare_b4d11a546d00b8add0ddb55324b3d490",
                        "VirusShare_b61fd6460dd70a9bf499f04234f74fb9", "VirusShare_b6361d512b0ca8cffa2a3c721c3a5950",
                        "VirusShare_b645937eebee1e0ae9fc73875fd07bf7", "VirusShare_b6611ff138818dda30caafe069c98020",
                        "VirusShare_b6a918f9f71f1e16de8e4a542612b719", "VirusShare_b82f8861f7ca6db5cdb0b5ba8cb20080",
                        "VirusShare_b85a2f2a315151e3c053fbb0a1f67350", "VirusShare_b87893ab3dc055c438e9204c25369260",
                        "VirusShare_b88459c6f6028f9fc846eb56d018087a", "VirusShare_b885b4ad01f226b4bdb4fafb3683d3c0",
                        "VirusShare_b8a28205659e1ee277617486ae0e788b", "VirusShare_b8ab4903421e2a26cc1e13919ba7c330",
                        "VirusShare_b8ae00a3edc6f1bfcdcf5f7284943a90", "VirusShare_b8db28cefb67f1a31fb3f56d55e887b0",
                        "VirusShare_b8de6af615b4fc21ddb9fc618d35ea0a", "VirusShare_ba033131f5f3f727b0603f5bb4983c63",
                        "VirusShare_ba1fdad4116bca2cb67da847f5fbfad0", "VirusShare_baa943bc5ab3c89a32d5c24167911b3e",
                        "VirusShare_bab181fb24dca67143f51085afabb41f", "VirusShare_baf63135d8b443a31bf0173396c2c7c0",
                        "VirusShare_bb14d20b4f276b6eb410bb2306a83500", "VirusShare_bb23e6ccc007e0af2288844f8794c79f",
                        "VirusShare_bb26f9aad0adddf614bddfd2fb7a36c2", "VirusShare_bb5c6269538552142d23c2d267f9248d",
                        "VirusShare_bb7113b321b15ade35be036591f49e20", "VirusShare_bb8a214271ea8f192f7bf8df279697da",
                        "VirusShare_bd12c98c5a1b89e9e850ff103cb07ea0", "VirusShare_bd13a57654c7ad9b98199533c2a99d80",
                        "VirusShare_bdb81be052693814cd7c502c99982890", "VirusShare_be1be025950d157b661c296c5757a410",
                        "VirusShare_be20d1d2a786898dd9a31a4fffaa88b4", "VirusShare_be570b4a6041d7b01d57e8c11d11bf35",
                        "VirusShare_bedc0d75e23842a192d932a1c1297cc4", "VirusShare_bef65bffc069ed73bb626b75cd5945ab",
                        "VirusShare_bf423b38e10f63e46c361ab289bbe373", "VirusShare_bf4244ae8e71ce8e34c8417e4de9f497",
                        "VirusShare_bf8b25d765e78a91f4449599f380d817", "VirusShare_bf8ff8ce94af18851dec844d755c96af",
                        "VirusShare_e0017891959d1b92a1621e1658092d4f", "VirusShare_e012c7c61ea87fa6e5c8a8bbf55095b0",
                        "VirusShare_e02383127df54877d193f7432f255568", "VirusShare_e0261409bc2c29650151aa8883f9171e",
                        "VirusShare_e064b6611d7991b199e22c5b16b03b60", "VirusShare_e0691927534f7ebee5511f6a96f0bc1f",
                        "VirusShare_e08b03dbadd96bedcfc64810dcb8a320", "VirusShare_e12ceee84b132ca85b733edefea54970",
                        "VirusShare_e17b488f186ee45c764977cd862846c0", "VirusShare_e242fad81c04c5478f89d03ce29989d9",
                        "VirusShare_e278be30795d8ce7f4bbb9c92ea9b300", "VirusShare_e2975974d94065a8b2e67863555aee3b",
                        "VirusShare_e2cfe83d6ab130d87d4ac061f7ae9a50", "VirusShare_e2d1a8d1ab737c2b8ac4bcbca344af38",
                        "VirusShare_e2e8dacb9991f0c33001f4d4f45352d0", "VirusShare_e2eae698f9da58b250d55b31b3353e50",
                        "VirusShare_e2ecc73e4c19d8d0221ca75bcfb22ad6", "VirusShare_e31927992c114ca1aef8bef03445cd7c",
                        "VirusShare_e31f94f6bdd5d8df18f920f85c7dd582", "VirusShare_e34b15d5bc9c2ef10183d1fa93771de0",
                        "VirusShare_e36caaa0848a1edc99b955ae6b5d8de0", "VirusShare_e3bd55dc9d312a1aeb3a7634822e4214",
                        "VirusShare_e3d2d1530b4f7b0215305b55b521fe40", "VirusShare_e3dec18df51866cb27ceb66ccf507be0",
                        "VirusShare_e4025d4674154ca913ec52d1cb6d8b5c", "VirusShare_e427fdc69146ab6049839fb8fa583240",
                        "VirusShare_e433422b3a51df9516e8b5c18c5b2510", "VirusShare_f084de9180eed45e9a3cb9dfdbb3e7c0",
                        "VirusShare_f0e994b2d3bd4688d0be214215bf7e50", "VirusShare_f0f1a0cf4a9bb6b4bf1c032aaa0dcb60",
                        "VirusShare_f0fa8b4a1de684d9baac322dc105171b", "VirusShare_f12e21b36c8af802af71b298fee65860",
                        "VirusShare_f14ff9623ce3b7b6f6c31589d4e8c6e8", "VirusShare_f16a0f729be04dbbfbcb56332ab2b1ff",
                        "VirusShare_f185a4c8b1763ca3310e1bcdc493b570", "VirusShare_f1871d5f3468d91a30ff5aa5794a5bf9",
                        "VirusShare_f18feaed155d59584057fd419e01b09c", "VirusShare_f1bd2d07695a946ba04c4cc64479cc60",
                        "VirusShare_f1be772ba1efa57a6a5fba113c21ad90", "VirusShare_f1cbfffc2380070b3561508125631e78",
                        "VirusShare_f1df3bac4a7d8a97fae03d84e1433b1c", "VirusShare_f1e5fe8524530ead674bcc2e12102657",
                        "VirusShare_30a335daac7b78cb1708190296f673f8", "VirusShare_31590d07e5572f727a2eb69d17705529",
                        "VirusShare_32993684ca07932e9598cc5d61f4f442", "VirusShare_35b99038e35fb7e3b0125abe896f6f8f",
                        "VirusShare_364139efc12bb8eaf06939c205c22f2b", "VirusShare_380abb82b4d43b7636d2da17bed3255c",
                        "VirusShare_399f7fc784782e2218c3cf2b10022c3e", "VirusShare_3e9fdc6b7214faa40d85290368fded4d",
                        "VirusShare_3eb8c93104ab082792e1a0e8c1b06962", "VirusShare_3f185b5688ca76febde11a43aa454f45",
                        "VirusShare_3fd3da747ac6e3383c322c70b1a276e0", "VirusShare_6afee5ef7dd79d5b811b89d70bf7bafb",
                        "VirusShare_702403793dc505992147718d88a30434", "VirusShare_7107d111c38ebe2ecef07c1a5fb94447",
                        "VirusShare_75e427653bc239fff64f2077d3cbd1b9", "VirusShare_76d472ccd387a4e5aa181b2661d7c800",
                        "VirusShare_7793b561f47cb3196db60d56a68925f9", "VirusShare_7a0541c28ca3ff844ee5ee0dc2f99d57",
                        "VirusShare_7a6faa5419f578e23c767bdb26e689bb", "VirusShare_7a79d16986131d7f288a33f9d39be9f3",
                        "VirusShare_7ad3263d4400d88cd2c87056faf49797", "VirusShare_7bf06654fcc6aa618f888b36e5703385",
                        "VirusShare_7c7310cae9f50910afb9b16dd8b3bdf4", "VirusShare_86887610a94158ac7ed0aff63a52b424",
                        "VirusShare_9b1fdfb5678a3e497a7b098fc7661800", "VirusShare_b10a4b6192641345afb1a265f44c3853",
                        "VirusShare_b128f223fa8287f859503c0028bf57e0", "VirusShare_b19ab7ba98abd7df1137171d6812ae94",
                        "VirusShare_b1baf7f3890ac27c5553b14507c204b0", "VirusShare_b38a38411a67dff1a9a10096036f84fc",
                        "VirusShare_b38df89c9bfc0d007eabaf5ce989e400", "VirusShare_b3c8c887b9e1259da406b7ac873a9247",
                        "VirusShare_b42efe75e0583f4afd14e9fa0bc372ba", "VirusShare_b43c31cac13d88137f5f17d3bb0c6c25",
                        "VirusShare_b44477da531e4e129b25f365a8c6ab9f", "VirusShare_b477f13b063d4fa6793f67fafe7f1790",
                        "VirusShare_b49bad926d1f687300fbee488b220687", "VirusShare_b4a9bbe7bdd233b691370bf486d2dda4",
                        "VirusShare_b4ac18a0afa828fb1053edbaf234e3c7", "VirusShare_b4d642bb1d26d9337866aa62ce8d8900",
                        "VirusShare_b4e246a93881b134dd9bf7326c1e9590", "VirusShare_b4e7aa470cabcb91a7e67e93355359f0",
                        "VirusShare_b4ed67692ac5f61c0a627fd724284ba0", "VirusShare_b61f3d2adc05f770bfe6000df87cace6",
                        "VirusShare_b62685ca6c4a55685f5fadff6e0d6870", "VirusShare_b63b0adf061c0eaac97600e1506f565c",
                        "VirusShare_b6594c3f6c60ce5b071730c6fe80114b", "VirusShare_b6f5de977b62793c2c5c036bd21b68f1",
                        "VirusShare_b80ac8d4c5ecad45fcd03eedc4012e40", "VirusShare_b837e166c1399d8ae2abee32ae3d7178",
                        "VirusShare_b86c029c1bf08c8826bdc61ba18f5b10", "VirusShare_b8f4ffcca5764d17c086806678fe45c9",
                        "VirusShare_ba159630b51a87bd3151838848d2221a", "VirusShare_ba4c9ab84714e79d571045a743d68dde",
                        "VirusShare_baabc9aa5c891d9215a1041f1c030bf0", "VirusShare_bae7c3b77df56d28da151b87fb0b8026",
                        "VirusShare_baf8e415cdc5ac00c227eec0d0430590", "VirusShare_bafa344724f00b25627e5f02831f5cf0",
                        "VirusShare_bb5bcb878ccd78ae14c3ede94e5bc2bb", "VirusShare_bbabf9626909f055e5407223c5f9cddd",
                        "VirusShare_bbbeaddf151542f6aa7db627e99a3a16", "VirusShare_bbce44a52088fc7fe8eca1d17df0a502",
                        "VirusShare_bbe63700772a558db165343c4069a1ef", "VirusShare_bd1181054c506d1e90ebe0d1a400b7b8",
                        "VirusShare_bd1400a25b50eed380257cbc60953da2", "VirusShare_bd31ff45c3b13ac78fe670467998dc2c",
                        "VirusShare_bd4a34170f4ac9620bb74856ae2d15e5", "VirusShare_bd6529e07104d6b1217a1ffc1807c060",
                        "VirusShare_bd82eb8daee5a5189d40bb8aa03d477d", "VirusShare_bd899462149e70fa99de6c5806055023",
                        "VirusShare_bd9a3b95d587b6242dc2ec2a2ce2ba85", "VirusShare_bdacf22b142b22de805350fa288fe6e3",
                        "VirusShare_bde758e1c8abc5b2cee8574505aaacd0", "VirusShare_be4d977479bbce4a8194ef58a23a3422",
                        "VirusShare_be6de6a5da172427ec5b2d614226fab5", "VirusShare_be9e6f1c3c50f261f815101157622c4c",
                        "VirusShare_bf040dfe7ab98f9a1d3e33f4351aa8cd", "VirusShare_bf9a3aa5557c66360bd90121129aaaea",
                        "VirusShare_bfd8e27ce04ca3f9c8b7e91ea4c0d1ab", "VirusShare_bfea82cd743d1b72ca098832500bda80",
                        "VirusShare_e04358b7d42438eac9f88e9c7f6c9f66", "VirusShare_e09c4422b26e79d4b00ddda64433f361",
                        "VirusShare_e0aa8eafdbc7d2f972426c31b60cc19b", "VirusShare_e11b2cfeaddae0e79fedd948a68b8fd4",
                        "VirusShare_e12428917f871f0a960bdaf0f793fffe", "VirusShare_e175232f05a52400be58907e5927ae3f",
                        "VirusShare_e178f993ee900e2df60499bba00d6017", "VirusShare_e1934abe6069b3b8704234b44a9ee1f6",
                        "VirusShare_e22024aead72f731cceae7bf383df9ca", "VirusShare_e221da7c78f288806869e92edf35394a",
                        "VirusShare_e235f0f87be25f3c2fb771ba30e36281", "VirusShare_e26076b48621d795a930e482009df6b8",
                        "VirusShare_e2bb45470bed441557e4f85c7e6b37b7", "VirusShare_e2dabd5239a13668224cd2af0b856010",
                        "VirusShare_e30686112afc44b1b9c565bbbbd6267a", "VirusShare_e31f168ba100bcd2aee7c52ea7d794c0",
                        "VirusShare_e32f5fc2677bc0be0058028605a2c2a0", "VirusShare_e356172164b60378f221c29d75477de9",
                        "VirusShare_e3657d375569b89f1a5ef58b014c2b66", "VirusShare_e376436dbf78da307f2b3b15ed712450",
                        "VirusShare_e3849f44f08e34351002e94e7b12d724", "VirusShare_e3c6680e504ff2019c840f2775173f28",
                        "VirusShare_e3de188f909da66354d7bea275038f67", "VirusShare_e3f1f6e926ddd5600b6854fc24bdbad0",
                        "VirusShare_e40b669bbd38583c4930095864de8510", "VirusShare_e44fe910fc606fda4013bc5887c9008e",
                        "VirusShare_f035aa98070be044dc385ef549f56210", "VirusShare_f08ceeefabacf1a939802f577a9d8c08",
                        "VirusShare_f0a7d8bd3439b72b1fd47a65e59c5b1c", "VirusShare_f0b86b260030df7c5076e5e39525f184",
                        "VirusShare_f0e03081e0dee7003901c9617fe29961", "VirusShare_f105474159e85cee905fac988e530f0f",
                        "VirusShare_f1095e3d25f2b1bef2986cdd9b9408df", "VirusShare_f17b4d25a480468dcb9cb6aa7cc88b10",
                        "VirusShare_f1da4e93df2acd1136f8dba51a3a905b", "VirusShare_f1dcc3958c858a61da62a6881c36e0fe",
                        "VirusShare_14be26c12262077c017a070e9b6e9457", "VirusShare_3038531478c111fe4a50ff2ee2be84d5",
                        "VirusShare_326f5a9d1d19885452284f0e1b8d298f", "VirusShare_3451433e5ecd135d4b7b5cedd0c6b091",
                        "VirusShare_3458495537a6fa09dd082d6bfe294af8", "VirusShare_361c82623959436eb802a41c9afd6cb0",
                        "VirusShare_38fc5350cd406e6d8eb22ad6d7567aed", "VirusShare_3a6413eee7292a2e426f6520c247e268",
                        "VirusShare_3dfcf9108ab5a0f715745d6f4550ed30", "VirusShare_3e86b747679a0fe3184fac4a538d48c6",
                        "VirusShare_54618b126c69b2f0a3309b7c0ac5ae26", "VirusShare_7149957243544b5b49cd82ea36ca9c40",
                        "VirusShare_7155d0360b602ae5a3fd6b9d8118dd10", "VirusShare_71b60dc7e7b00cb40c78eab9d82618d0",
                        "VirusShare_71fb2a7e238a8cae8824965f481671da", "VirusShare_7528b1cc5945718836ceef8d885fa6e0",
                        "VirusShare_790e8000534dfd6654a9e21b79d3b09e", "VirusShare_7b34f009293a18e1fafedc01a88867db",
                        "VirusShare_7b4acb10c7dde8620dba87e8490e1cd2", "VirusShare_7cad18c796b72137984ed777fa1b9dc0",
                        "VirusShare_b123b6ccff1ef7d4905207d4f412ccc0", "VirusShare_b14b3028fe7b573cfff23f1d916c5a96",
                        "VirusShare_b32fa7a903f9bd01880c535ad317c7c6", "VirusShare_b34417995ce1c8596aaa596c58b82e40",
                        "VirusShare_b3476ee283e9c25b83fe5821b6317918", "VirusShare_b3534c713115751260873cbac39b20ad",
                        "VirusShare_b3624472c1b911bd80ae86e43360721d", "VirusShare_b36a75a10eb6e253ef9312bd8deb6f9d",
                        "VirusShare_b3806c5719556e8f74536ed969ca76d0", "VirusShare_b3a43dd11bc7d6747af4bf9f12f2daf7",
                        "VirusShare_b41e490fff89caec43d4b4d899223eb0", "VirusShare_b43aa191f9168dc5627c437044b6e960",
                        "VirusShare_b4805524e1ecba58e162119529ff80f0", "VirusShare_b4b69859b9637ce3f774a0ff493ad160",
                        "VirusShare_b4ef8569046bbe57e3dd103068d7783f", "VirusShare_b61df949b1a600d7b1408461983e94e0",
                        "VirusShare_b6243abbbcfc7ab357a1dcccada4c540", "VirusShare_b6901f646da9f29e4045c167cfda28f1",
                        "VirusShare_b6ee5d2262b3cef3eef138c3b31e0d9c", "VirusShare_b8056b292f2497856080340604b8de44",
                        "VirusShare_b83d965f1f9922ff330024163253ae20", "VirusShare_b870711879e1aa08ab3b6786fbe9c4a0",
                        "VirusShare_b8c646cc4397cc0e876fae32c6e55bc0", "VirusShare_b8d4804f64c89d285478d61d503a1f2a",
                        "VirusShare_b8d9a14ab9e40ff2927e193fc98f5326", "VirusShare_b8f7ac3c6e68233f68f58c54efa7def0",
                        "VirusShare_ba4b5c936b30ec35654015f1947ef04c", "VirusShare_ba5a4ed7f53fc4c6657acd3cf98ad874",
                        "VirusShare_bac56ffbbc45a80396dc2c6cb3aa8980", "VirusShare_bad2d9b6bf63c3b99b472e4ae9162300",
                        "VirusShare_bbd2c7098a6015d505c23673b4fa4620", "VirusShare_bbf0be0063d8f940418531121158b85f",
                        "VirusShare_bbf817a2e9714c39dcef1c65955a3f40", "VirusShare_bd26ddc401772a573763c5013f676f70",
                        "VirusShare_bd39da927981d746ee9f95f09187a420", "VirusShare_bd4ab788ed1cfed1244033c967aaff2d",
                        "VirusShare_bd67015467e453329d7c4f617ee55950", "VirusShare_bd751cf193001d006514d646f9850800",
                        "VirusShare_bdd32245e0c0ec89205800ed2bbfa690", "VirusShare_bdeb033e7e07843263bcec073dcb4d20",
                        "VirusShare_be4d328be61dcfaf4f0488c62911d9d0", "VirusShare_be5d44f826f7192b9e0a9ce3e0b7b604",
                        "VirusShare_be7c0e2c8a3c4fe2aefb6edc8b2738bc", "VirusShare_be7e4d23bb6cd6b577f1653152a052e0",
                        "VirusShare_bea67122cc17d8ee220f9ad60cef3717", "VirusShare_bed19bd5c27d30c486ba08b9445e33c4",
                        "VirusShare_bede8cfa6364d703df52143dc0095ca4", "VirusShare_bf136adc5cb1f63c1b794287076a022f",
                        "VirusShare_bf5997f20dd2817539a9abe88cc23750", "VirusShare_bf9c1b1fcf8ad692daa049510b0ffa4f",
                        "VirusShare_bfa30f4c43add682c712bef5b6fc1c16", "VirusShare_bfa44f055af17e975ed22ca22ca49060",
                        "VirusShare_bfc6335e3ea32351ec9f21d78b0836b3", "VirusShare_bfd8388d8ebf54e081aa6c304b11bf80",
                        "VirusShare_bfdc25c3b3872953870409737e124fe0", "VirusShare_bfe76a6d3b1d4aecc6a6cfe0bc9231db",
                        "VirusShare_e02878648bae848bec183b2dbdd7e990", "VirusShare_e02e9a5567b04827e3f92d5b210d14a4",
                        "VirusShare_e040d270101b8036887c24cee2921ba1", "VirusShare_e05a8f47e065e734084de69ec2fbdd50",
                        "VirusShare_e062b8a7d915615dba6717958fb67780", "VirusShare_e06d569238cb6f98fe36a277ed621080",
                        "VirusShare_e070845d94c412cb5d1d18eb3a366721", "VirusShare_e07e14012f99eac5eb81cc2083592f00",
                        "VirusShare_e0a4c55da7cfbd590f1f4d661b5f2960", "VirusShare_e0b19987edca0a8f611138c64b88322c",
                        "VirusShare_e0bd5c88352a566a226be71f67e658d0", "VirusShare_e10ad0f3e8b10b851af44b1284a2021b",
                        "VirusShare_e120c0156a191c9f379773e659608ae8", "VirusShare_e121136251e0d6f5963b5f44e83a05a9",
                        "VirusShare_e1505ab739a2cb4dfe8c9f54359f0457", "VirusShare_e16be681a705c13f4ae24eb0d22dc020",
                        "VirusShare_e175149022c17026d77195f5025ced60", "VirusShare_e19ad1caaf533477627ea4c81079f1f3",
                        "VirusShare_e1a728d1e65403ea7f2b8c2bd203f7e0", "VirusShare_e1f3fc08627c0f11f1fcb8a3e52f3b13",
                        "VirusShare_e22b8a6d597c9e0004b0a8e2cb12ce81", "VirusShare_e29b72f03f4c04f6cb1e6dcdd02aa162",
                        "VirusShare_e2af70d6ada3259759f9636a2ce2c898", "VirusShare_e30fe17d41b94239e5e4f7f4b55c5710",
                        "VirusShare_e3150922849203b98860393254cc169d", "VirusShare_e322f6655057f4b6615bd9b027e0e3a0",
                        "VirusShare_e37c838c14d7506706ad2294f45cdadc", "VirusShare_e37d65e2b8beb0d36cd07d2dee643e56",
                        "VirusShare_e397e53a62e1b0dcb63e8bbe986cad70", "VirusShare_e3a6269e4116a0f1e90d506e605e3dc0",
                        "VirusShare_e3f011a1fdff969135e9d24c93eea8ba", "VirusShare_e3f036b57db8044a8885569b9bc75b0b",
                        "VirusShare_e4049f5fc822dbb23f10625ca5f37166", "VirusShare_f00d4050f78cd63166ff3bb9b18b222b",
                        "VirusShare_f029f3401eb07ee7d49b1e94fb28c7d5", "VirusShare_f0615a4d9c6b532fda275c6d9e657145",
                        "VirusShare_f075dc8e49da24f9c5b0ca3452ed2db0", "VirusShare_f07bd8bb8e619b2d18dc55876027d994",
                        "VirusShare_f141d1b2e8c3bf92190965d41fb103e0", "VirusShare_f162e5a31d1c176fe0e7252b5b094fde",
                        "VirusShare_f1773bb0203cb72ebce268e50e876e8b"]

        # self.parser.parse_each_document(list_of_docs, coll)
        # hcp = self.parser.convert2vec()
        # input_matrix = np.array(hcp)
        # pi.dump(input_matrix, open("input_matrix.dump", "w"))

        input_file = pi.load(open("/home/satwik/Documents/Research/MachineLearning/Miscellaneous/input_matrix.dump"))

        input_matrix = list(list())
        for each in input_file:
            input_matrix.append(list(each))

        input_matrix = np.array(input_matrix, dtype=float)

        # sim, dist = self.pairwise_jaccard(input_matrix)
        dist = self.pairwise_euclidean(input_matrix)
        dist = self.pairwise_cosine(input_matrix)
        dist = self.pairwise_hamming(input_matrix)
        # clusters = self.get_clusters(sim, 0.9)
        # print clusters

        plt.hist(dist)
        plt.savefig("distribution.png", format='eps', dpi=1000)
        print("Done")
        # plt.show()


if __name__ == "__main__":
    pjs = PairwiseDistanceMeasures()
    pjs.main()