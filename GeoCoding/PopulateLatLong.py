from urllib.parse import quote
from json import load

from HelperFunctions.HelperFunction import HelperFunction
from GeoCoding.IPDataApi import IPDataApi

from Utils.LoggerUtil import LoggerUtil
from Utils.DBUtils import DBUtils
from Utils.ConfigUtil import ConfigUtil


class PopulateLatLong:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.db_utils = DBUtils()
        self.config = ConfigUtil.get_config_instance()
        self.helper = HelperFunction()
        self.ip_data = IPDataApi()

    def get_collection(self):
        username = self.config['environment']['mongo']['username']
        pwd = self.config['environment']['mongo']['password']
        password = quote(pwd)
        address = self.config['environment']['mongo']['address']
        port = self.config['environment']['mongo']['port']
        auth_db = self.config['environment']['mongo']['auth_db']
        is_auth_enabled = self.config['environment']['mongo']['is_auth_enabled']

        client = self.db_utils.get_client(address=address, port=port, auth_db=auth_db,
                                          is_auth_enabled=is_auth_enabled,
                                          username=username, password=password)
        db_name = self.config['environment']['mongo']['db_name']
        cuckoo_db = client[db_name]

        c2db_collection_name = self.config['environment']['mongo']['c2db_collection_name']
        c2db_collection = cuckoo_db[c2db_collection_name]

        lat_long_collection_name = self.config['environment']['mongo']['lat_long_collection']
        lat_long_collection = cuckoo_db[lat_long_collection_name]

        ip_collection_name = self.config['environment']['mongo']['ip_collection']
        ip_collection = cuckoo_db[ip_collection_name]

        return c2db_collection, lat_long_collection, ip_collection

    @staticmethod
    def get_query(lok):
        query = [
            {"$match": {"key": {"$in": lok}, "feature": "network"}},
            {"$addFields": {"__order": {"$indexOfArray": [lok, "$key"]}}},
            {"$sort": {"__order": 1}}
        ]
        return query

    def get_ip_list(self, cursor):
        meta_list = list()
        for each_doc in cursor:
            try:
                feature_values = list()
                key = each_doc["key"]
                feature_values += each_doc["value"][key]["network"]["domains"]["ip"]
                feature_values += each_doc["value"][key]["network"]["hosts"]["ip"]
                feature_values += each_doc["value"][key]["network"]["udp"]["dst"]
                feature_values += each_doc["value"][key]["network"]["udp"]["src"]
                feature_values = list(filter(None, set(feature_values)))
                meta_list.append(feature_values)
            except Exception as e:
                self.log.error("Error : {}".format(e))
        meta_list = self.helper.flatten_list(meta_list)
        return meta_list

    def fill_db(self, lat_long_collection, ip_collection, ip_dict, lat_long_dict):
        try:
            ip_collection.insert_one(ip_dict)
            lat_long_collection.insert_one(lat_long_dict)
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def main(self):
        families_path = self.config["data"]["malware_families_list"]
        fam_lok_dict = load(open(families_path + "/" + "family_to_key_list_mapping.json", "r"))

        c2db_collection, lat_long_collection, ip_collection = self.get_collection()

        for family, lok in fam_lok_dict.items():
            lok = self.helper.convert_to_vs_keys(lok)
            query = self.get_query(lok=lok)
            cursor = c2db_collection.aggregate(query)
            list_op_ips = self.get_ip_list(cursor)
            ip_dict, lat_long_dict = self.ip_data.get_ip_data(ip_addr_list=list_op_ips)
            self.fill_db(lat_long_collection, ip_collection, ip_dict, lat_long_dict)


if __name__ == '__main__':
    populate = PopulateLatLong()
    populate.main()
