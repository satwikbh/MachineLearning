import ast
import json
import os
import pickle as pi
import time
from collections import defaultdict

from Clustering import Clustering


class Validation:
    def __init__(self):
        pass

    def insert_data_into_dict(self, key, each_file, file_path, counter):
        f = open(file_path + "/" + each_file)
        s = ""

        for lines in f.readlines():
            s += lines

        s = dict(ast.literal_eval(s))
        temp1 = dict()
        temp1["positives"] = s.get("positives")
        temp1["total"] = s.get("total")
        scans = s.get("scans")

        temp1 = defaultdict(list)
        for each_av in scans:
            try:
                temp1[each_av].append(scans.get(each_av).get("result"))
            except Exception as e:
                print("The error is : {}, \n caused by : {}".format(e, each_av))
        temp2 = self.meta_cluster.get(key)
        if counter != 0:
            finaldict = dict()
            for stupid_key in temp1:
                try:
                    temp3 = list()
                    temp3 += temp1[stupid_key]
                    temp3 += temp2[stupid_key]
                    finaldict[stupid_key] = temp3
                except Exception as e:
                    print("The error is : {}, \n caused by : {}".format(e, stupid_key))
            self.meta_cluster[key] = finaldict
        else:
            self.meta_cluster[key] = temp1
            print("Bye")

    def create_folders_dict(self, folder_names):
        self.meta_cluster = dict()
        for each in folder_names:
            self.meta_cluster[each] = defaultdict(list)

    @staticmethod
    def get_path():
        return os.path.abspath(os.path.dirname("__file__"))

    def main(self, labels):
        path = self.get_path()
        start_time = time.time()

        self.create_folders_dict(labels.keys())

        x = 0
        for key, value in labels.items():
            dir_path = path + "/" + "cluster_" + key + "/"
            files = os.listdir(dir_path)
            for each_file in files:
                self.insert_data_into_dict(key, each_file, dir_path, x)
                x += 1

        pi.dump(self.meta_cluster, open(path + "/" + "final_clusters.dump", "w"))
        with open('FinalResult.json', 'w') as outfile:
            json.dumps(self.meta_cluster, outfile)
        print("Total time taken : {}".format(time.time() - start_time))


if __name__ == "__main__":
    clus_val = Clustering()
    clus_val.initial_check(clus_val.set_path())
    validate = Validation()
    validate.main(clus_val.labels)
