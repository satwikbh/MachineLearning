from json import dumps

from Cluster2DB.PerformAnalysis import PerformAnalysis
from Utils.LoggerUtil import LoggerUtil
from Utils.KafkaUtil import KafkaUtil
from Utils.ConfigUtil import ConfigUtil


class CuckooQueue:
    """
    This class will queue the executable for behavioral analysis. This is done using Kafka.
    The producer produces the message which contains Queue class object and puts it in the channel.
    Once the executable is analyzed, the result success or failed will be appended in the respective collection.
    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.queue = Queue()
        self.cuckoo_object = PerformAnalysis()

    def producer(self, kafka, topic_name, md5_value, md5_path):
        """
        This method adds the executable and its path to the queue collection.
        Behavioral analysis is performed for elements in this collection.
        :param kafka:
        :param topic_name:
        :param md5_value:
        :param md5_path:
        :return:
        """
        meta_list = self.queue.insert_in_queue(md5_value, md5_path)
        message = dumps(meta_list)
        producer = kafka.get_producer()
        producer.send(topic_name, value=message)

    def main(self, md5_value, md5_path):
        broker_list = self.config["environment"]["kafka"]["broker_list"]
        topic_name = self.config["environment"]["kafka"]["broker_list"]
        kafka = KafkaUtil(broker_list=broker_list)
        self.producer(kafka, topic_name, md5_value, md5_path)


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
