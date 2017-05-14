import time
from pymongo import MongoClient
import vtlite
import urllib


class VirusTotal:
    def __init__(self):
        pass

    @staticmethod
    def main():
        client = MongoClient(
            "mongodb://" + "localhost:27017" + "/" + "admin")
        db = client["cuckoo"]
        vt_collection = db["VirusTotalResults"]
        malware_collection = db["cluster2db"]

        list_of_md5 = malware_collection.find({"key": {"$exists": True}}).distinct("key")

        print("Total number of malware's are : {}".format(len(list_of_md5)))

        for index, each_malware_md5 in enumerate(list_of_md5):
            if "VirusShare_" in each_malware_md5:
                vtlite.main(vt_collection, each_malware_md5.split("_")[1])
                print("Malware {} report generated".format(index))
                time.sleep(15)
            else:
                continue


if __name__ == '__main__':
    vt = VirusTotal
    vt.main()
