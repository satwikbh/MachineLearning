import pickle as pi
import time
from collections import defaultdict
from subprocess import check_output

import numpy as np
import os
from sklearn.cluster import DBSCAN


class Clustering:
    def __init__(self):
        self.names = list()
        self.labels = defaultdict(list)

    def prepare_names_list(self, path):
        f1 = open(path + "/" + "mycsvfile.csv")
        for lines in f1.readlines():
            self.names.append(lines.split(",")[0])

    def prepare_labels_list(self, path):
        f = open(path + "/" + "reduced_matrix.dump")
        l = list(list())

        for lines in f.readlines():
            temp = list()
            for here in lines.split(","):
                if '[' in here:
                    here = here.replace('[', '')
                if ']' in here:
                    here = here.replace(']', '')
                temp.append(float(here))
            l.append(temp)

        input_matrix = np.array(l)

        dbscan = DBSCAN().fit(input_matrix)

        d = dbscan.labels_.tolist()

        for key, value in enumerate(d):
            self.labels[str(value)].append(self.names[key])

    def initial_check(self, path):
        if os.path.exists(path + "/" + "names.dump"):
            self.names = pi.load(open(path + "/" + "names.dump"))
        else:
            self.prepare_names_list(path)
        if os.path.exists(path + "/" + "labels.dump"):
            self.labels = pi.load(open(path + "/" + "labels.dump"))
        else:
            self.prepare_labels_list(path)
            pi.dump(self.labels, open(path + "/" + "labels.dump", "w"))

    @staticmethod
    def set_path():
        _current_dir = os.path.abspath(os.path.dirname("__file__"))
        _path = "/".join(_current_dir.split("/")[:-1])
        _path += "/" + "cluster"
        return _path

    @staticmethod
    def create_virus_total_json(each, key):
        try:
            md5 = each.split("_")[1]
            check_output(["python", "vtlite.py", md5, "-svj"])
            file_name = "VTDL" + md5.upper() + ".json"
            if not os.path.isdir("cluster_" + str(key)):
                check_output(["mkdir", "cluster_" + str(key)])
            check_output(["mv", file_name, "cluster_" + str(key) + "/"])
            time.sleep(15)
        except Exception as e:
            print(e, "key : {}, \n, value : {}".format(key, each))

    def main(self):
        path = self.set_path()
        self.initial_check(path)

        for key, value in self.labels.items():
            if not os.path.exists("cluster_" + str(key)):
                for each in value:
                    self.create_virus_total_json(each, key)


if __name__ == '__main__':
    clus_val = Clustering()
    clus_val.main()
