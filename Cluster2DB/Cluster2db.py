import pickle as pi
import time
from os.path import isfile, join

import os
from pymongo import MongoClient


class Cluster2db(object):
    """docstring for cluster2db."""

    @staticmethod
    def get_client():
        # return MongoClient(
        #     "mongodb://" + "admin" + ":" + urllib.quote("goodDeveloper@123") + "@" + "localhost:27017" + "/" + "admin")
        return MongoClient(
            "mongodb://localhost:27017" + "/" + "admin")

    @staticmethod
    def get_collection(client):
        db = client.get_database("cuckoo")
        collection = db.get_collection("cluster2db")
        return collection

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

    @staticmethod
    def convert_behavior_dump_to_json(value):
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
            print(e)

    @staticmethod
    def convert_network_inner_dicts(inner_dict):
        """

        :param value:
        :return:
        """
        for key, value in inner_dict.items():
            if isinstance(value, set):
                inner_dict[key] = list(inner_dict[key])
        return inner_dict

    def convert_network_dump_to_json(self, value):
        """
        Here the Key will be network.
        The inner keys of the json are 'domains', 'udp', 'hosts', 'dns'. Further the inner_list contain the values which are in set format.
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
            print(e)

    @staticmethod
    def convert_static_dump_to_json(value):
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
            print(e)

    @staticmethod
    def convert_statistics_dump_to_json(value):
        """

        :param value:
        :return:
        """
        try:
            value["statSignatures"] = list(value.get("statSignatures"))
        except Exception as e:
            print(e)

    @staticmethod
    def convert_signatures_dump_to_json(value):
        """
        Mongo doesn't accept int keys, this method will convert them to string and also makes the flatten list out of the values.
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
            print(e)
        main_doc['signatures'] = doc
        return main_doc

    @staticmethod
    def convert_unknown_features_to_json(value):
        """
        
        :param value: 
        :return: 
        """
        main_doc = dict()
        try:
            pass
        except Exception as e:
            print(e)
        return main_doc

    @staticmethod
    def convert_malheur_value_to_json(value):
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
            print(e)
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

            if isinstance(behavior_profile, list):
                if fname.endswith(".failedAnalyses.cluster"):
                    document = dict()
                    document["key"] = "failedAnalyses"
                    cursor = collection.find({"key": "failedAnalyses"})
                    failed_list = list()
                    for each in cursor:
                        failed_list += each.get("value")
                    failed_list += behavior_profile
                    document["value"] = list(set(failed_list))
                    collection.remove({"key": "failedAnalyses"})
                    collection.insert(document)

            if isinstance(behavior_profile, dict):
                if len(behavior_profile.keys()) <= 0:
                    return
                else:
                    md5 = behavior_profile.keys()[0]
                    document = dict()
                    if '.' in md5:
                        behavior_profile[md5.split(".")[0]] = behavior_profile.pop(md5)
                        document['key'] = md5.split(".")[0]
                    else:
                        document['key'] = md5
                    value = behavior_profile.values()[0]
                    if fname.endswith(".behaviorDump.cluster"):
                        self.convert_behavior_dump_to_json(value)
                        document['feature'] = 'behavior'
                    elif fname.endswith(".networkDump.cluster"):
                        self.convert_network_dump_to_json(value)
                        document['feature'] = 'network'
                    elif fname.endswith(".staticDump.cluster"):
                        self.convert_static_dump_to_json(value)
                        document['feature'] = 'static'
                    elif fname.endswith(".statsDump.cluster"):
                        self.convert_statistics_dump_to_json(value)
                        document['feature'] = 'statSignatures'
                    elif fname.endswith(".signatureDump.cluster"):
                        behavior_profile = {document['key']: self.convert_signatures_dump_to_json(value)}
                        document['feature'] = 'signatures'
                    elif fname.endswith(".malheurDump.cluster"):
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
            print(e)

    @staticmethod
    def key_mapping(key):
        if key == "statSignatures":
            return "statsDump"
        if key == "static":
            return "staticDump"
        if key == "signatures":
            return "signatureDump"
        if key == "network":
            return "networkDump"
        if key == "malheur":
            return "malheurDump"
        if key == "behavior":
            return "behaviorDump"

    @staticmethod
    def present_in_db(collection, files_list):
        list_of_keys_in_mongo = list()
        cursor = collection.find({"key": {"$exists": True}}, ["key", "feature"])
        for each in cursor:
            key = each.get("key")
            feature = each.get("feature")
            if "failedAnalyses" not in key:
                val = key + "." + Cluster2db.key_mapping(feature) + ".cluster"
                list_of_keys_in_mongo.append(val)
        return set(files_list).difference(set(list_of_keys_in_mongo))

    def main(self):
        print("Process started")

        start_time = time.time()
        collection = self.get_collection(self.get_client())
        print("Enter the path of the clusters : ")
        # path = str(raw_input())
        path = "/Users/satwik/Documents/IIIT/MS_Thesis/Cluster/cluster/"
        files_list = os.listdir(path)
        print("Total number of files in cluster: {}".format(len(files_list)))

        updated_list = self.present_in_db(collection, files_list)
        print("Total number of files already in mongo: {}".format(len(files_list) - len(updated_list)))

        bulk = collection.initialize_unordered_bulk_op()

        for each in updated_list:
            if isfile(join(path, each)):
                self.dump_to_document(path + each, collection, bulk)
        try:
            bulk.execute()
        except Exception as e:
            print(e)
        end_time = time.time()

        print("Process done")

        print("Total time taken : {}".format(
            end_time - start_time))


if __name__ == '__main__':
    clus2db = Cluster2db()
    clus2db.main()
