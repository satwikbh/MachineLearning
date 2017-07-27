import urllib
import pickle as pi
import hickle as hkl
import numpy as np
from collections import defaultdict
from scipy.sparse import hstack, vstack

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from HelperFunctions.HelperFunction import HelperFunction


class MalheurBasedClasses:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.config = ConfigUtil().get_config_instance()
        self.db_utils = DBUtils()

    def get_collection(self):
        username = self.config['environment']['mongo']['username']
        pwd = self.config['environment']['mongo']['password']
        password = urllib.quote(pwd)
        address = self.config['environment']['mongo']['address']
        port = self.config['environment']['mongo']['port']
        auth_db = self.config['environment']['mongo']['auth_db']
        is_auth_enabled = self.config['environment']['mongo']['is_auth_enabled']

        client = self.db_utils.get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=is_auth_enabled,
                                          username=username, password=password)

        db_name = self.config['environment']['mongo']['db_name']
        db = client[db_name]
        collection_name = self.config['environment']['mongo']['collection_name']
        collection = db[collection_name]
        return client, collection

    def create_labels(self, cursor, labelled_data_path):
        md5_class = defaultdict(list)
        class_md5 = defaultdict(list)

        for each in cursor:
            if each['feature'] == "malheur":
                key = each['key']
                family = each['value'][key]['malheur']['family']
                md5_class[key].append(family)
                class_md5[family].append(key)

        class_md5["UNCLASSIFIED"] = class_md5.get("")
        class_md5.pop("")

        self.helper.create_dir_if_absent(labelled_data_path)
        list_of_classes = set(class_md5.keys())
        file_names = list()
        for each_malware_class in list_of_classes:
            file_name = labelled_data_path + "/" + each_malware_class + ".dump"
            hkl.dump(np.asarray(class_md5.get(each_malware_class)), open(file_name, "w"))
            file_names.append(file_name)

        hkl.dump(md5_class, open(labelled_data_path + "all_md5_classes.metadump", "w"))
        hkl.dump(class_md5, open(labelled_data_path + "all_classes_md5.metadump", "w"))
        return file_names

    @staticmethod
    def get_list_of_keys(file_names):
        list_of_keys = list()
        for each_file in file_names:
            if "UNCLASSIFIED" not in each_file:
                l = hkl.load(open(each_file)).tolist()
                list_of_keys += l
        return list_of_keys

    def main(self):
        labelled_data_path = self.config['classifier_data']['labelled_data']
        client, collection = self.get_collection()
        names = pi.load(open(self.config['data']['list_of_keys'] + "/" + "names.dump", "w"))
        cursor = collection.aggregate([{"$group": {"_id": names}}])

        file_names = self.create_labels(cursor, labelled_data_path)
        list_of_keys = self.get_list_of_keys(file_names)
