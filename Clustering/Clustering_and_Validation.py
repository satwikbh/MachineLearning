import ast
import os
import pickle as pi
import time
from collections import defaultdict
from subprocess import check_output

import numpy as np
from sklearn.cluster import DBSCAN


class Clustering:
    def __init__(self):
        self.names = list()
        self.labels = defaultdict(list)

    f = open("reduced_matrix")
    f1 = open("mycsvfile.csv")

    def prepare_names_list(self):
        for lines in self.f1.readlines():
            self.names.append(lines.split(",")[0])

    def prepare_labels_list(self):
        l = list(list())

        for lines in self.f.readlines():
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

    def main(self):
        for key, value in self.labels.items():
            for each in value:
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


class Validation:
    def __init__(self):
        self.k7antivirus = defaultdict(list)
        self.kaspersky = defaultdict(list)
        self.fsecure = defaultdict(list)
        self.trend_micro = defaultdict(list)
        self.mcafee_gw_edition = defaultdict(list)
        self.sophos = defaultdict(list)
        self.mcafee = defaultdict(list)
        self.panda = defaultdict(list)
        self.total_defense = defaultdict(list)
        self.quickheal = defaultdict(list)
        self.malwarebytes = defaultdict(list)
        self.k7gw = defaultdict(list)
        self.symantec = defaultdict(list)
        self.eset = defaultdict(list)
        self.trend_micro_house_call = defaultdict(list)
        self.avast = defaultdict(list)
        self.clamav = defaultdict(list)
        self.bit_defender = defaultdict(list)
        self.comodo = defaultdict(list)
        self.avira = defaultdict(list)
        self.microsoft = defaultdict(list)
        self.avg = defaultdict(list)

    def switch_replica(self, argument):
        self.final_av_list = ["K7AntiVirus", "Kaspersky", "F-Secure", "TrendMicro", "McAfee-GW-Edition", "Sophos",
                              "McAfee", "Panda", "TotalDefense", "CAT-QuickHeal", "Malwarebytes", "K7GW",
                              "Symantec", "ESET-NOD32", "TrendMicro-HouseCall", "Avast", "ClamAV", "BitDefender",
                              "Comodo", "Avira", "Microsoft", "AVG"]
        switcher = {
            "K7AntiVirus": self.k7antivirus,
            "Kaspersky": self.kaspersky,
            "F-Secure": self.fsecure,
            "TrendMicro": self.trend_micro,
            "McAfee-GW-Edition": self.mcafee_gw_edition,
            "Sophos": self.sophos,
            "McAfee": self.mcafee,
            "Panda": self.panda,
            "TotalDefense": self.total_defense,
            "CAT-QuickHeal": self.quickheal,
            "Malwarebytes": self.malwarebytes,
            "K7GW": self.k7gw,
            "Symantec": self.symantec,
            "ESET-NOD32": self.eset,
            "TrendMicro-HouseCall": self.trend_micro_house_call,
            "Avast": self.avast,
            "ClamAV": self.clamav,
            "BitDefender": self.bit_defender,
            "Comodo": self.comodo,
            "Avira": self.avira,
            "Microsoft": self.microsoft,
            "AVG": self.avg,
        }
        return switcher.get(argument, "nothing")

    def some_function(self, path, each_file):
        f = open(path + "/" + file)
        s = ""
        for lines in f.readlines():
            s += lines

        s = dict(ast.literal_eval(s))
        temp = dict()
        temp["positives"] = s.get("positives")
        temp["total"] = s.get("total")
        scans = s.get("scans")
        for each_av in scans:
            self.switch_replica(each_av).append(scans.get(each_av).get("result"))

    def main(self, labels, path):
        start_time = time.time()
        for key, value in labels.items():
            dir_path = path + "/" + "cluster_" + key + "/"
            files = os.listdir(dir_path)
            for each_file in files:
                self.some_function(each_file, path)

        final_result = list()
        for each in self.final_av_list:
            final_result.append(self.switch_replica(each))

        pi.dump(final_result, open(path + "final_clusters", "w"))
        print("Total time taken : {}".format(time.time() - start_time))


if __name__ == '__main__':
    clus_val = Clustering()
    clus_val.main()

    validate = Validation()
    validate.main(clus_val.labels, "")
