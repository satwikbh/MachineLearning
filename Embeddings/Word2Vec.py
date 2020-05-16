from urllib.parse import quote
from gensim.models import Word2Vec, FastText, word2vec

from PrepareData.ParsingLogic import ParsingLogic
from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils


class Word2Vec:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.db_utils = DBUtils()
        self.parser = ParsingLogic(use_trie_pruning=False)

    def get_collection(self):
        username = self.config['environment']['mongo']['username']
        pwd = self.config['environment']['mongo']['password']
        password = quote(pwd)
        address = self.config['environment']['mongo']['address']
        port = self.config['environment']['mongo']['port']
        auth_db = self.config['environment']['mongo']['auth_db']
        is_auth_enabled = self.config['environment']['mongo']['is_auth_enabled']

        client = self.db_utils.get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=is_auth_enabled,
                                          username=username, password=password)

        db_name = self.config['environment']['mongo']['db_name']
        db = client[db_name]

        c2db_collection_name = self.config['environment']['mongo']['c2db_collection_name']
        c2db_collection = db[c2db_collection_name]

        return c2db_collection

    def main(self):
        c2db_collection = self.get_collection()
        list_of_keys = ["VirusShare_332538300b0c76b3fba5b5ad1ca07c30", "VirusShare_db62d4df458cb7249ad4961a76285ca0",
                        "VirusShare_35699e4901933a47fb03ca119cb56cc0", "VirusShare_37cdb9ad374d502c34aad61bb6823e10",
                        "VirusShare_3589cd62d38663732cb1678cb7e97890", "VirusShare_7d0921eda6356511c914ee94cfbe02f0",
                        "VirusShare_e038a8474dea642035b03c401dc346d0", "VirusShare_be763e0397810f7afdc566f42d2ba140",
                        "VirusShare_bade06b967e569276f816ea12d021a20", "VirusShare_cc256f9420a9a23dfa03149ad1a95520",
                        "VirusShare_f0426a50f1d1e5cbfd4024e2cf3a5260", "VirusShare_378a7db088cc465e57f09051ad73afb0",
                        "VirusShare_31d520d02f4bc2e487e2ef1052019a10", "VirusShare_3ed1b9555b788b1c179f7b08428d5b30",
                        "VirusShare_b8ee631c0e46c6839f4b81dff642d3f0", "VirusShare_b31ddf00996218a8307d567c753f6f80",
                        "VirusShare_77a17681f1114cbaebcb77dd25648c20", "VirusShare_bb0a694bee0fdbe7be10e19924f94980",
                        "VirusShare_f1634b6518d096650c4354b78fa0a4b0", "VirusShare_dc0c71443270e71c9a3dcad68da1c910",
                        "VirusShare_38d45d22c9a283346b71b006256cecd0", "VirusShare_f13b53bfa2fd53d00adcbf6719d38410",
                        "VirusShare_3987ed498e0f0893b84f5bc290193ca0", "VirusShare_bf90beb19c4100c469617d13eddd9540",
                        "VirusShare_336bef6ed1648ac67d512e127b3c8e70", "VirusShare_bde8ec3c925852106abc353a457bb1e0",
                        "VirusShare_ba0a12d577cf4a15fc800b0e39e857b0", "VirusShare_35d112e1fa2c6d1be7d309c78436f290",
                        "VirusShare_ba1a9546188128084ea9046bce7f8260", "VirusShare_38ce036aa23610b38eb25b63ecbdef60",
                        "VirusShare_b6a268b6ecbf7c1bca3a6f3399370910", "VirusShare_b410f138a98046f9a105407e76811f90",
                        "VirusShare_361ecb69bff6a4184cbfcb7029fc5010", "VirusShare_3afc4bdd63ed24bd9f57e795bbd730f0",
                        "VirusShare_3d42d4edf17591988fb4fa38a3b09cc0", "VirusShare_bb393d1533e58189fcd8086d2cc52d10",
                        "VirusShare_b87e2624d3b38963454420706bf1cde0", "VirusShare_b641d61aa4b1c79948bbc14d7db3ed00",
                        "VirusShare_351402128a5d48da92b150b3c46b2750", "VirusShare_327d6485643e8bea1d75161fd03ef780",
                        "VirusShare_3b1af254ca2119ccd2ffe793c7affc80", "VirusShare_38c9f73bacb0be0a2762ef7683e4c160",
                        "VirusShare_33e77fb5a377857e0197d80ec121eca0", "VirusShare_360fd547853e7a9ac96468d520858960",
                        "VirusShare_b6c0321e005867c9991b331d10f4d690", "VirusShare_bf76f9364bc527bb42d1934cc4f53ac0",
                        "VirusShare_ba50e0f3a9a519ba177d015319215500", "VirusShare_e038d3db8d5f9ca6ad3d0b4f8a15df20",
                        "VirusShare_3aa4b5f0826666d8ac349e03a896c260", "VirusShare_b6dced02c62a37a2ca0b999327cfea00",
                        "VirusShare_e01cedc16939d115c1dd8085671c79a0", "VirusShare_ba6856289166d8249283359cec4dc830",
                        "VirusShare_3015628dc529a93906a0ef84b21ec090", "VirusShare_33da3fefe6028665363ce243e00edc80",
                        "VirusShare_ba2d111716d09b6fc67a304be0f35d80", "VirusShare_e009e99d34c28ee331eee1c414362b00",
                        "VirusShare_ba8acbfc1f0cdd4f3ebdf3d8cbb852e0", "VirusShare_b8907dd83d6e1caf7217d1ef33651000",
                        "VirusShare_cbda431bf0b30b33cf1063408406bc80", "VirusShare_f1c34e46a70857c5714b4efd995af990",
                        "VirusShare_b88fd62063cc7edac2f6a67bcd175890", "VirusShare_e00ec108290002e73ba99e90d7925560",
                        "VirusShare_b83943b9beb9edd9710f8b86072e1b30", "VirusShare_36f5e56c5490bd98a3744a76205f60a0",
                        "VirusShare_7cb3b1823ccd57d48889c12ca0ccfbd0", "VirusShare_baa4fd02370c8d3d909fc6e6581408e0",
                        "VirusShare_b4e64e75665f61e1024278dc965fff30", "VirusShare_b89b450e5dd537a49bf0f0764514c3e0",
                        "VirusShare_3bbdb8609d56f977c27ff278f49536d0", "VirusShare_e01e7152e2df502bf5800c0e824f16a0",
                        "VirusShare_3ae504662749dc0d5e41651663d69ef0", "VirusShare_b8ae706de4527a9f22b27cc35389dcf0",
                        "VirusShare_776691e169dda68a5d73741f7ba28ca0", "VirusShare_b6eccf43181a4b992e724a5806d10ba0",
                        "VirusShare_b6f72bb00726ca367d125a927beafc40", "VirusShare_bb564247f88c7a14c311b421691fc8d0",
                        "VirusShare_b8acae5a8f144a4c26ff77daafd284d0", "VirusShare_cb769b2a97172465fb2dbfaf5b43d140",
                        "VirusShare_b853da36ef0fc8426592c8ebf4dc0540", "VirusShare_ba6f9c9a108721493961629eaeab6ac0",
                        "VirusShare_31fe6cd2b3bce265f1c7336ce3db4b90", "VirusShare_b8b8d504f2b73c9235117203a08fe660",
                        "VirusShare_3b258bd54cdaaa6d9cb5c987a23017a0", "VirusShare_3f2fb84271dd7d5cd745ea1d0f21a9d0",
                        "VirusShare_bdfea0f69213027bdf41d88d742a1650", "VirusShare_31b8418721aeeed691ab823aaaabc0d0",
                        "VirusShare_3aaf34697353374c8c21f3353570d610", "VirusShare_3ccca26cb7899fd5c524fa4608e65eb0",
                        "VirusShare_bb6840b5eeabf9f815401017795dedb0", "VirusShare_ba95019f810ec47a62bedd89c4f6caa0",
                        "VirusShare_b60d74137dd8d895578e733a322f31e0", "VirusShare_39c1254b5dc8107b782bf864d4710220",
                        "VirusShare_72e203c8ae9372e152fabac32e4674e0", "VirusShare_bb7aebe25425a194c2778425cff17b80",
                        "VirusShare_3b32f4b2d50b0ec87a73526f03bf6ad0", "VirusShare_340ed801725f236952e7d2f15d991320",
                        "VirusShare_b6dc3a2fbc6eb7581623a4a53c00a320", "VirusShare_e0230fa68d921796ca18618b82d4e550",
                        "VirusShare_bb0a9a00c1cae8c76c0be6b5ba413000", "VirusShare_3a708cd3cc91ad9b6acb77de3095c380"]
        doc2bow = self.parser.parse_each_document(list_of_keys, c2db_collection)
        # print(type(doc2bow))
        model = word2vec.Word2Vec(sentences=doc2bow.values(), min_count=2, size=30, compute_loss=True)
        print(F"Training Loss : {model.get_latest_training_loss()}")
        word = "stealth_network".lower()
        p = model.wv[word]
        print(model.batch_words)
        print(F"Similar : {model.wv.similar_by_word(word=word, topn=5)}")
        print(F"Word : {p}")


if __name__ == '__main__':
    w2v = Word2Vec()
    w2v.main()
