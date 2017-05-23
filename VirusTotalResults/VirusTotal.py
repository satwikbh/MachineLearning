import time
from pymongo import MongoClient
import vtlite
import urllib


class VirusTotal:
    def __init__(self):
        pass

    @staticmethod
    def query_and_insert_into_db(list_of_md5, vt_collection):
        for index, each_malware_md5 in enumerate(list_of_md5):
            if "VirusShare_" in each_malware_md5:
                counter = index % 2
                vtlite.main(vt_collection, each_malware_md5.split("_")[1], counter)
                print("Malware {} report generated".format(index))
                if counter == 0:
                    time.sleep(7)
                else:
                    time.sleep(8)
            else:
                continue

    @staticmethod
    def main():
        client = MongoClient(
            "mongodb://" + "localhost:27017" + "/" + "admin")
        db = client["cuckoo"]
        vt_collection = db["VirusTotalResults"]
        malware_collection = db["cluster2db"]

        already_scanned_md5 = vt_collection.find({"malware_source": {"$exists": True}}).distinct("malware_source")
        to_scan_list_of_md5 = malware_collection.find({"key": {"$exists": True}}).distinct("key")

        list_of_md5 = list(set(to_scan_list_of_md5) - set(already_scanned_md5))

        print("Total number of malware's are : {}".format(len(list_of_md5)))

        while True:
            VirusTotal.query_and_insert_into_db(list_of_md5, vt_collection)
            updated_list = vt_collection.find({}).distinct("malware_source")
            list_of_md5 = malware_collection.find({"key": {"$exists": True}}).distinct("key")
            list_of_md5 = list(set(list_of_md5) - set(updated_list))
            if len(list_of_md5) < 1:
                break

        print("Mission Impossible is completed !!!!")


if __name__ == '__main__':
    vt = VirusTotal
    vt.main()
