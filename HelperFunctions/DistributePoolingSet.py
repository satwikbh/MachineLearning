import hickle
import numpy as np
import json

from Utils.LoggerUtil import LoggerUtil


class DistributePoolingSet:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    def distributed_pool(self, mini_batch, part):
        try:
            self.log.info("Storing the {} part of the object".format(part))
            fname = "feature_pool_part-" + str(part) + ".hkl"
            json_dump_value = json.dumps(mini_batch)
            hickle.dump(json_dump_value, fname, mode='w', compression='gzip')
            return fname
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def distributed_feature_vector(self, mini_batch, part):
        try:
            self.log.info("Storing the {} part of the object".format(part))
            fname = "feature_vector_part-" + str(part) + ".hkl"
            numpy_dense_array = np.asarray(mini_batch.todense())
            hickle.dump(numpy_dense_array, fname, mode='w', compression='gzip')
            return fname
        except Exception as e:
            self.log.error("Error : {}".format(e))
