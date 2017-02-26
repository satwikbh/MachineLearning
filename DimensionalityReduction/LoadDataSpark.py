import json
import logging.config
import os
import pickle as pi
import time

import numpy as np
from pyspark import SparkContext

sc = SparkContext('local', 'pyspark')


class LoadData:
    logger = logging.getLogger(__name__)

    def __init__(self):
        pass

    @staticmethod
    def setup_logging(default_path='logging.json', default_level=logging.INFO, env_key='LOG_CFG'):
        """
        Setup logging configuration
        :param default_path:
        :param default_level:
        :param env_key:
        :return:
        """
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)

    @staticmethod
    def create_dumps(data_path, input_matrix, names_list):
        LoadData.logger.info("Entering the {} class".format(LoadData.create_dumps.__name__))
        LoadData.logger.info("Creating the Dumps")

        pi.dump(input_matrix, open(data_path + "/" + "input_matrix", "w"))
        pi.dump(names_list, open(data_path + "/" + "names_list", "w"))

        LoadData.logger.info("Completed creating the Dumps")
        LoadData.logger.info("Entering the {} class".format(LoadData.create_dumps.__name__))

    @staticmethod
    def load_data(data_path):
        LoadData.logger.info("Entering the {} class".format(LoadData.load_data.__name__))
        rdd = sc.textFile(data_path + "/" + "mycsvfile.csv")
        length = rdd.map(lambda line: line.split(",")[1][:-2]).filter(lambda line: list(line)).count()
        l = list(list())

        names = rdd.map(lambda line: line.split(",")[0]).take(length)
        temp_list = rdd.map(lambda line: line.split(",")[1][:-2]).take(length)

        for each in temp_list:
            l.append(list(each))

        input_matrix = np.array(l)
        names = np.array(names)

        LoadData.create_dumps(data_path, input_matrix, names)
        LoadData.logger.info("Exiting the {} class".format(LoadData.load_data.__name__))

    def main(self):
        start_time = time.time()
        LoadData.logger.info("Entering the {} class".format(LoadData.main.__name__))
        print("Enter the path to load data from")
        data_path = str(raw_input())
        self.load_data(data_path)
        print("Creating data is completed.")
        LoadData.logger.info("Exiting the {} class".format(LoadData.main.__name__))
        LoadData.logger.info("Total time taken : {}".format(time.time() - start_time))


if __name__ == '__main__':
    ld = LoadData()
    ld.setup_logging()
    ld.main()
