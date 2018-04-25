import time
from collections import defaultdict

import falconn as falc
import numpy as np

from Utils import LoggerUtil


class FalconnLsh:
    def __init__(self):
        self.log = LoggerUtil.LoggerUtil(self.__class__.__name__).get()

    @staticmethod
    def set_lsh_parameters(dataset):
        lsh_params = falc.LSHConstructionParameters()

        lsh_params.dimension = len(dataset[1])
        lsh_params.lsh_family = 'cross_polytope'

        # TODO : Can the below distance function be NegativeInnerProduct instead of EuclideanSquared??
        lsh_params.distance_function = 'euclidean_squared'

        lsh_params.storage_hash_table = 'bit_packed_flat_hash_table'

        # If we cannot determine the number of threads setting it ONE is an ideal way for it to infer.
        # The number of threads used is always at most the number of tables l.
        lsh_params.num_setup_threads = 1
        lsh_params.num_rotations = 2
        lsh_params.l = 125

        # we build 20-bit hashes so that each table has
        # 2^20 bins; this is a good choise since 2^20 is of the same
        # order of magnitude as the number of data points
        # falc.compute_number_of_hash_functions(20, lsh_params)
        falc.compute_number_of_hash_functions(8, lsh_params)
        index = falc.LSHIndex(lsh_params)

        return index

    def get_clusters(self, index, dataset, threshold):
        temp_pool = range(len(dataset))
        d = defaultdict(list)
        count = 0
        while len(set(temp_pool)) > 1:
            try:
                if temp_pool[count] == -1:
                    count += 1
                    continue
                value = index.find_near_neighbors(dataset[temp_pool[count]], threshold=threshold)
                if len(value) == 1:
                    key = 'cluster_' + str(count)
                    d[key].append(temp_pool[count])
                    temp_pool[count] = -1
                    count += 1
                if len(value) > 1:
                    key = 'cluster_' + str(count)
                    for each in value:
                        d[key].append(each)
                        temp_pool[each] = -1
                    count += 1
            except Exception as e:
                self.log.error(e, temp_pool, count, exc_info=True)
        return d

    def lsh_falconn(self, inp, threshold):
        self.log.info("************ FALCONN *************")
        start_time = time.time()
        # Converting the type as float32. This is what the falconn package expects.
        dataset = inp.astype(np.float32)

        # Centering the data
        dataset -= np.mean(dataset, axis=0)

        # Using default parameters for now. Need to change this as per out requirement.
        # params = falc.get_default_parameters(dimension=dataset.shape[1], num_points=dataset.shape[0])

        # Custom parameters for LSH
        index = self.set_lsh_parameters(dataset)
        index.setup(dataset=dataset)

        d = self.get_clusters(index, dataset, threshold)
        self.log.info("Time taken for FALCONN : {}".format(time.time() - start_time))
        return d
