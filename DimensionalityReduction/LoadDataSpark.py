import pickle as pi
import time

import numpy as np
from pyspark import SparkContext

from Utils.LoggerUtil import LoggerUtil

sc = SparkContext('local', 'pyspark')


class LoadData:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    def create_dumps(self, data_path, input_matrix, names_list):
        self.log.info("Entering the {} class".format(LoadData.create_dumps.__name__))
        self.log.info("Creating the Dumps")

        pi.dump(input_matrix, open(data_path + "/" + "input_matrix", "w"))
        pi.dump(names_list, open(data_path + "/" + "names_list", "w"))

        self.log.info("Completed creating the Dumps")
        self.log.info("Entering the {} class".format(LoadData.create_dumps.__name__))

    def load_data(self, data_path):
        self.log.info("Entering the {} class".format(LoadData.load_data.__name__))
        rdd = sc.textFile(data_path + "/" + "mycsvfile.csv", 50)
        length = rdd.map(lambda line: line.split(",")[1][:-2]).filter(lambda line: list(line)).count()
        l = list(list())

        names = rdd.map(lambda line: line.split(",")[0]).take(length)
        value = rdd.map(lambda line: line.split(",")[1][:-2]).take(length)

        for each in value:
            l.append(list(each))

        input_matrix = np.array(l)
        names = np.array(names)

        self.create_dumps(data_path, input_matrix, names)
        self.log.info("Exiting the {} class".format(LoadData.load_data.__name__))

    def main(self):
        start_time = time.time()
        self.log.info("Entering the {} class".format(LoadData.main.__name__))
        print("Enter the path to load data from")
        data_path = str(raw_input())
        self.load_data(data_path)
        print("Creating data is completed.")
        self.log.info("Exiting the {} class".format(LoadData.main.__name__))
        self.log.info("Total time taken : {}".format(time.time() - start_time))


if __name__ == '__main__':
    ld = LoadData()
    ld.main()
