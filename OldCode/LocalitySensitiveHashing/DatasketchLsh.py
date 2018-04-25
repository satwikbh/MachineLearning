import time

from datasketch import MinHash, MinHashLSH

from Utils import LoggerUtil


class DatasketchLsh:
    def __init__(self):
        self.log = LoggerUtil.LoggerUtil(self.__class__.__name__).get()

    def lsh_datasketch(self, inp, threshold, num_perm):
        self.log.info("************ DATASKETCH *************")
        start_time = time.time()
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        d = dict()
        cluster = dict()

        for x in xrange(len(inp)):
            d[x] = MinHash(num_perm=num_perm)

        for index, value in enumerate(inp):
            d[index].update(value)
            lsh.insert("m" + str(index), d[index])

        for key, value in d.items():
            cluster[key] = lsh.query(value)

        self.log.info("Time taken for DATASKETCH : {}".format(time.time() - start_time))
        return cluster
