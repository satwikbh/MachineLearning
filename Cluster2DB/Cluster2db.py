import json
import mmap
import os
import pickle as pi
import time
import urllib
from os.path import isfile, join

from pymongo import MongoClient


class Cluster2db(object):
    """docstring for cluster2db."""

    @staticmethod
    def get_client():
        return MongoClient(
            "mongodb://" + "cuckoo" + ":" + urllib.quote("goodDeveloper@123") + "@" + "localhost:27017" + "/" + "admin")

    @staticmethod
    def get_collection(client):
        db = client.get_database("cuckoo")
        collection = db.get_collection("cluster2db")
        return collection

    @staticmethod
    def insert_one_by_one(collection, gfs):
        with open('mycsvfile.csv', 'rb') as f:
            # Size 0 will read the ENTIRE file into memory!
            m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

            while True:
                try:
                    input_dictionary = dict()
                    data = m.readline()
                    key, value = data.split(",")
                    input_dictionary[key] = list(value[:-2])
                    # collection.insert_one(input_dictionary)
                    gfs.put(str(json.dumps(input_dictionary)))
                except Exception as e:
                    print e, key
                    break

    @staticmethod
    def bulk_insert(collection):
        with open('mycsvfile.csv', 'rb') as f:
            # Size 0 will read the ENTIRE file into memory!
            m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            l = list()
            while True:
                try:
                    input_dictionary = dict()
                    data = m.readline()
                    key, value = data.split(",")
                    input_dictionary[key] = list(value[:-2])
                    l.append(input_dictionary)
                    if (len(l) % 500 == 0):
                        collection.insert_many(l)
                        l = list()
                except Exception as e:
                    print e, key
                    break

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

    def dump_to_document(self, fname, collection):
        """
        This method will parse the pickle file and then dumps them into Mongo.
        :return:
        """
        try:
            f = open(fname)
            d = pi.load(f)
            md5 = d.keys()[0]
            document = dict()
            if '.' in md5:
                d[md5.split(".")[0]] = d.pop(md5)
                document['key'] = md5.split(".")[0]
            else:
                document['key'] = md5
            value = d.values()[0]
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
            else:
                return

            document['value'] = d
            collection.insert_one(document)
        except Exception as e:
            print(e)

    def main(self):
        print("Process started")

        start_time = time.time()
        collection = self.get_collection(self.get_client())
        print("Enter the path of the clusters : ")
        path = str(raw_input())
        files_list = os.listdir(path)
        for each in files_list:
            if isfile(join(path, each)):
                self.dump_to_document(path + each, collection)
        end_time = time.time()

        print("Process done")

        print("Total time taken : {}".format(
            end_time - start_time))


if __name__ == '__main__':
    clus2db = Cluster2db()
    clus2db.main()
