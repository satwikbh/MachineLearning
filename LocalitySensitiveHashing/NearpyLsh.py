import time

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

import HelperFunctions.helper_functions as utils
from Utils import LoggerUtil


class NearpyLsh:
    def __init__(self):
        self.log = LoggerUtil.LoggerUtil(self.__class__.__name__).get()

    def nearpy_lsh(self, inp):
        self.log.info("************ NEARPY *************")
        start_time = time.time()
        dimension = inp.shape[1]
        rbp = RandomBinaryProjections('rbp', utils.HelperFunction.get_nearest_power_of_two(dimension))
        engine = Engine(dimension, lshashes=[rbp])

        clusters = dict()

        for index, value in enumerate(inp):
            engine.store_vector(value, 'data_%d' % index)

        for index, value in enumerate(inp):
            try:
                engine.store_vector(value, 'data_%d' % index)
                val = engine.neighbours(value)
                clusters[index] = []
                if len(val) > 1:
                    temp = []
                    for each in val:
                        temp.append(each[1])
                    clusters[index] += temp
                else:
                    clusters[index] = [val[0][1]]
            except Exception as e:
                self.log.error(e, index, value, exc_info=True)
        self.log.info("Time taken for NEARPY : {}".format(time.time() - start_time))
        return clusters
