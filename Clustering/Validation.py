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
        current_dict = dict()
        current_dict["positives"] = s.get("positives")
        current_dict["total"] = s.get("total")
        scans = s.get("scans")

        current_dict = defaultdict(list)
        for each_av in scans:
            try:
                current_dict[each_av].append(scans.get(each_av).get("result"))
            except Exception as e:
                print("The error is : {}, \n caused by : {}".format(e, each_av))
        meta_dict = self.meta_cluster.get(key)
        if counter != 0:
            finaldict = dict()
            for stupid_key in current_dict:
                try:
                    intermediate_list = list()
                    if current_dict.has_key(stupid_key):
                        intermediate_list += current_dict[stupid_key]
                    else:
                        intermediate_list.append(None)
                    if meta_dict.has_key(stupid_key):
                        intermediate_list += meta_dict[stupid_key]
                    else:
                        intermediate_list.append(None)
                    finaldict[stupid_key] = intermediate_list
                except Exception as e:
                    print("Error when Merging the current dict with Meta dict. \n Error Caused by : {}".format(e))
            self.meta_cluster[key] = finaldict
        else:
            self.meta_cluster[key] = current_dict

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
        with open(path + "/" + "FinalResult.json", 'w') as outfile:
            json.dump(self.meta_cluster, outfile)
        print("Total time taken : {}".format(time.time() - start_time))


if __name__ == "__main__":
    clus_val = Clustering()
    clus_val.initial_check(clus_val.set_path())
    validate = Validation()
    validate.main(clus_val.labels)
