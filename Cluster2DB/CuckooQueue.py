from Utils.LoggerUtil import LoggerUtil
from PerformAnalysis import PerformAnalysis


class CuckooQueue:
    """
    This class will queue the executable for behavioral analysis. This is done using Kafka.
    The producer produces the message which contains Queue class object and puts it in the channel.
    Once the executable is analyzed, the result success or failed will be appended in the respective collection.
    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.queue = Queue()
        self.cuckoo_object = PerformAnalysis()

    def producer(self, md5_value, md5_path):
        """
        This method adds the executable and its path to the queue collection.
        Behavioral analysis is performed for elements in this collection.
        :param md5_value:
        :param md5_path:
        :return:
        """
        meta_list = self.queue.insert_in_queue(md5_value, md5_path)
        # TODO : Insert this meta_list item in the channel so that PerformAnalysis class consumer can consume it.

    def main(self, md5_value, md5_path):
        self.producer(md5_value, md5_path)


class Queue:
    def __init__(self):
        pass

    def insert_in_queue(self, md5_value, md5_path):
        """
        Given a md5 value, this method creates a custom object, inserts the value and returns it.
        :param md5_path:
        :param md5_value:
        :return:
        """
        meta_list = list()
        element = self.create_object(md5_value, md5_path)
        meta_list.append(element)
        return meta_list

    @staticmethod
    def create_object(md5_value, md5_path):
        element = dict()
        element["md5"] = md5_value
        element["fil_path"] = md5_path
        element["success"] = 0
        element["failure"] = 0
        return element


if __name__ == '__main__':
    cuckoo_queue = CuckooQueue()
    cuckoo_queue.main(md5_path=None, md5_value=None)
