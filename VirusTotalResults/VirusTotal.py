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
            "mongodb://" + "admin" + ":" + urllib.quote("goodDevelopers@123") + "@" + "localhost:27017" + "/" + "admin")
        db = client["cuckoo"]
        vt_collection = db["VirusTotalResults"]
        malware_collection = db["cluster2db"]

        list_of_md5 = malware_collection.find({"key": {"$exists": True}}).distinct("key")

        print("Total number of malware's are : {}".format(len(list_of_md5)))

        for index, each_malware_md5 in enumerate(list_of_md5):
            vtlite.main(vt_collection, each_malware_md5)
            print("Malware {} report generated".format(index))
            time.sleep(15)


if __name__ == '__main__':
    vt = VirusTotal
    vt.main()
