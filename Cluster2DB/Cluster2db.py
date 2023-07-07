import pickle as pi
import os

from urllib.parse import quote
from time import time
from os.path import isfile, join

from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from Utils.LoggerUtil import LoggerUtil


class Cluster2db(object):
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.db_utils = DBUtils()

    def get_collection(self):
        username = self.config['environment']['mongo']['username']
        pwd = self.config['environment']['mongo']['password']
        address = self.config['environment']['mongo']['address']
        port = self.config['environment']['mongo']['port']
        auth_db = self.config['environment']['mongo']['auth_db']
        is_auth_enabled = self.config['environment']['mongo']['is_auth_enabled']

        password = quote(pwd)

        client = self.db_utils.get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=is_auth_enabled,
                                          username=username, password=password)

        db_name = self.config['environment']['mongo']['db_name']
        db = client[db_name]

        c2db_collection_name = self.config['environment']['mongo']['c2db_collection_name']
        c2db_collection = db[c2db_collection_name]

        return c2db_collection

    @staticmethod
    def flatten_list(nested_list):
        flattened_list = list()
        for sublist in nested_list:
            if isinstance(sublist, list):
                for item in sublist:
                    flattened_list.append(item)
            else:
                flattened_list.append(sublist)
        return flattened_list

    def convert_behavior_dump_to_json(self, value):
        """
        Here the Key will be behavior.
        The inner keys of the json are 'files', 'keys', 'summary', 'mutexes', 'executed_commands'.
        This method will convert the sets of these keys into list so that they can be converted to JSON.
        :param value:
        :return behavior:
        """
        try:
            behavior = value.get("behavior")
            for key, value in behavior.items():
                if isinstance(value, set):
                    behavior[key] = list(behavior[key])
        except Exception as e:
            self.log.error("Error : {}".format(e))

    @staticmethod
    def convert_network_inner_dicts(inner_dict):
        """

        :param inner_dict:
        :return:
        """
        for key, value in inner_dict.items():
            if isinstance(value, set):
                inner_dict[key] = list(inner_dict[key])
        return inner_dict

    def convert_network_dump_to_json(self, value):
        """
        Here the Key will be network.
        The inner keys of the json are 'domains', 'udp', 'hosts', 'dns'.
        Further the inner_list contain the values which are in set format.
        This method will convert the sets of these keys into list so that they can be converted to JSON.
        :param value:
        :return behavior:
        """
        try:
            network = value.get("network")
            for key, value in network.items():
                if isinstance(value, set):
                    network[key] = list(network[key])
                if isinstance(value, dict):
                    network[key] = self.convert_network_inner_dicts(value)
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def convert_static_dump_to_json(self, value):
        """

        :param value:
        :return:
        """
        try:
            static = value.get("static")
            for key, value in static.items():
                if isinstance(value, set):
                    static[key] = list(static[key])
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def convert_statistics_dump_to_json(self, value):
        """

        :param value:
        :return:
        """
        try:
            value["statSignatures"] = list(value.get("statSignatures"))
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def convert_signatures_dump_to_json(self, value):
        """
        Mongo doesn't accept int keys, this method will convert them to string.
        Also makes the flatten list out of the values.
        :param value: 
        :return: 
        """
        main_doc = dict()
        doc = dict()
        try:
            signatures = value.get('signatures')
            for key, value in signatures.items():
                doc[str(key)] = list(set(Cluster2db.flatten_list(value)))
        except Exception as e:
            self.log.error("Error : {}".format(e))
        main_doc['signatures'] = doc
        return main_doc

    @staticmethod
    def convert_unknown_features_to_json(value):
        """
        TODO: Need to implement this.
        :param value: 
        :return: 
        """
        main_doc = dict()
        return main_doc

    def convert_malheur_value_to_json(self, value):
        """
        Convert the malheur result into a json with family and score key's
        :param value: 
        :return: 
        """
        main_doc = dict()
        doc = dict()
        doc['family'] = None
        doc['score'] = None
        try:
            malheur = value.get("malheur")
            doc['family'] = malheur[0]
            doc['score'] = malheur[1]
        except Exception as e:
            self.log.error("Error : {}".format(e))
        main_doc['malheur'] = doc
        return main_doc

    def dump_to_document(self, fname, collection, bulk):
        """
        This method will parse the pickle file and then dumps them into Mongo.
        :return:
        """
        try:
            f = open(fname)
            behavior_profile = pi.load(f)

            if isinstance(behavior_profile, list) and fname.endswith(".failed_analyses.cluster"):
                document = dict()
                document["key"] = "failed_analyses"
                cursor = collection.find({"key": "failed_analyses"})
                failed_list = list()
                for each in cursor:
                    failed_list += each.get("value")
                failed_list += behavior_profile
                document["value"] = list(set(failed_list))
                collection.remove({"key": "failed_analyses"})
                collection.insert(document)

            if isinstance(behavior_profile, dict):
                if len(behavior_profile.keys()) <= 0:
                    return
                else:
                    md5 = [_ for _ in behavior_profile.keys()][0]
                    document = dict()
                    if '.' in md5:
                        behavior_profile[md5.split(".")[0]] = behavior_profile.pop(md5)
                        document['key'] = md5.split(".")[0]
                    else:
                        document['key'] = md5
                    value = [_ for _ in behavior_profile.values()][0]
                    if fname.endswith(".behavior_dump.cluster"):
                        self.convert_behavior_dump_to_json(value)
                        document['feature'] = 'behavior'
                    elif fname.endswith(".network_dump.cluster"):
                        self.convert_network_dump_to_json(value)
                        document['feature'] = 'network'
                    elif fname.endswith(".static_dump.cluster"):
                        self.convert_static_dump_to_json(value)
                        document['feature'] = 'static'
                    elif fname.endswith(".statsDump.cluster"):
                        self.convert_statistics_dump_to_json(value)
                        document['feature'] = 'statSignatures'
                    elif fname.endswith(".signature_dump.cluster"):
                        behavior_profile = {document['key']: self.convert_signatures_dump_to_json(value)}
                        document['feature'] = 'signatures'
                    elif fname.endswith(".malheur_dump.cluster"):
                        behavior_profile = {document['key']: self.convert_malheur_value_to_json(value)}
                        document['feature'] = 'malheur'
                    elif fname.endswith(".unknownFeatures.cluster"):
                        behavior_profile = {document['key']: self.convert_unknown_features_to_json(value)}
                        document['feature'] = 'unknownFeatures'
                    else:
                        return

                    document['value'] = behavior_profile
                    bulk.insert(document)

        except Exception as e:
            self.log.error("Error : {}".format(e))

    @staticmethod
    def key_mapping(key):
        if key == "statSignatures":
            return "statsDump"
        if key == "static":
            return "static_dump"
        if key == "signatures":
            return "signature_dump"
        if key == "network":
            return "network_dump"
        if key == "malheur":
            return "malheur_dump"
        if key == "behavior":
            return "behavior_dump"

    @staticmethod
    def present_in_db(collection, files_list):
        list_of_keys_in_mongo = list()
        cursor = collection.find({"key": {"$exists": True}}, ["key", "feature"])
        for each in cursor:
            key = each.get("key")
            feature = each.get("feature")
            if "failed_analyses" not in key:
                val = key + "." + Cluster2db.key_mapping(feature) + ".cluster"
                list_of_keys_in_mongo.append(val)
        return set(files_list).difference(set(list_of_keys_in_mongo))

    def main(self):
        start_time = time()
        c2db_collection = self.get_collection()
        cluster_path = self.config['environment']['cluster_path']
        files_list = os.listdir(cluster_path)
        self.log.info("Path of the clusters : {}".format(cluster_path))
        self.log.info("Total number of files in clusters: {}".format(len(files_list)))

        updated_list = self.present_in_db(c2db_collection, files_list)
        self.log.info("Total number of files already in mongo: {}".format(len(files_list) - len(updated_list)))

        bulk = c2db_collection.initialize_unordered_bulk_op()

        for index, each in enumerate(updated_list):
            if index % 1000 == 0:
                self.log.info("Iteration : #{}".format(index / 1000))
            if isfile(join(cluster_path, each)):
                self.dump_to_document(cluster_path + each, c2db_collection, bulk)
        try:
            bulk.execute()
        except Exception as e:
            self.log.error("Error : {}".format(e))

        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    clus2db = Cluster2db()
    clus2db.main()
