import urllib
import numpy as np
import pickle as pi

from collections import defaultdict

from Utils.LoggerUtil import LoggerUtil
from Utils.DBUtils import DBUtils
from DimensionalityReduction.PcaNew import PcaNew
from HelperFunctions.ParsingLogic import ParsingLogic


class PrepareDataset:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.db_utils = DBUtils()
        self.parser = ParsingLogic()
        self.dim_red = PcaNew()

    def get_collection(self):
        username = "admin"
        password = urllib.quote("goodDevelopers@123")
        address = "localhost"
        port = "27017"
        auth_db = "admin"

        client = self.db_utils.get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=False,
                                          username=None, password=None)

        db = client['cuckoo']
        collection = db['cluster2db']
        return collection

    def get_families_data(self, collection, list_of_keys):
        entire_families = defaultdict(list)

        for each_key in list_of_keys:
            query = {"feature": "malheur", "key": each_key}
            local_cursor = collection.find(query)
            for each_value in local_cursor:
                key = each_value['key']
                entire_families[each_value['value'][key]["malheur"]["family"]].append(key)

        entire_families.pop("")
        self.log.info("Total Number of families : {} ".format(len(entire_families)))

    def get_data_as_matrix(self, collection, list_of_keys):
        pi.dump(list_of_keys, open("names.dump", "w"))
        self.parser.parse_each_document(list_of_keys, collection)
        hcp = self.parser.convert2vec()
        input_matrix = np.array(hcp)
        pi.dump(input_matrix, open("input_matrix.dump", "w"))
        input_file = list()
        for each in input_matrix:
            input_file.append(list(each))
        input_matrix = np.array(input_file, dtype=float)
        return input_matrix

    def load_data(self):
        collection = self.get_collection()
        cursor = collection.aggregate([{"$group": {"_id": '$key'}}])
        list_of_keys = list()

        for each_element in cursor:
            list_of_keys.append(each_element['_id'])

        self.get_families_data(collection, list_of_keys)
        input_matrix = self.get_data_as_matrix(collection, list_of_keys)
        self.dim_red.dimensionality_reduction(input_matrix)


if __name__ == "__main__":
    prepare_dataset = PrepareDataset()
    prepare_dataset.load_data()
