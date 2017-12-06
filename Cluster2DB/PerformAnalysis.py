import json

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from Utils.KafkaUtil import KafkaUtil


class PerformAnalysis:
    """
    This class invokes the scripts neccessary for the behavioral analysis of the malware.
    The malware executable's path is picked from the collection analysis is done.
    It appends to the success or failure collections based on the execution status.
    This class is Asynchronous and doesn't respond to any class.
    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()

    def submit_2_cuckoo(self, file_path):
        """
        The executable is submitted to cuckoo sandbox for dynamic analysis using the file_path.
        It returns success or failure based on cuckoo's response.
        :param file_path:
        :return:
        """
        # TODO : Need to complete this method.
        if True:
            return "success"
        else:
            return "failure"

    def to_book_keeping_db(self, md5_value, status, book_keeping_db):
        """
        Inserts a new malware into success or failure collection based on the status.
        In case the malware is new then it doesn't exist.
        If analysis succeeds then value is 1.
        If analysis fails then increment the failure value.
        :param md5_value:
        :param status:
        :param book_keeping_db:
        :return:
        """
        if status == "success":
            success_collection = book_keeping_db["success"]
            doc = dict()
            doc["md5_value"] = md5_value
            doc["success_count"] = 1
            success_collection.insert_one(doc)
        else:
            failure_collection = book_keeping_db["failure"]
            cursor = failure_collection.find({"md5_value": md5_value})
            if cursor.count() > 0:
                value = cursor["failure_count"] + 1
                self.log.info("Malware with md5 : {} failed for the {} time".format(md5_value, cursor["failure_count"]))
                failure_collection.update_one({"md5_value": md5_value}, {'$push': {'failure_count': value}})
            else:
                doc = dict()
                doc["md5_value"] = md5_value
                doc["failure_count"] = 1
                failure_collection.insert_one(doc)

    def consume(self, book_keeping_db, kafka, topic_name):
        """
        This consumer consumes from the channel and gets the executable's md5 and its file path.
        Once the analysis is done,
        :param book_keeping_db:
        :param topic_name:
        :param kafka:
        :return:
        """
        self.log.info("Consuming from the topic name : {}".format(topic_name))
        consumer = kafka.get_consumer()
        consumer.subscribe(topics=topic_name)
        for message in consumer:
            try:
                self.log.info("Consumer Details : \ntopic : {}, partition : {}, offset : {}".format(message.topic,
                                                                                                    message.partition,
                                                                                                    message.offset))
                value = message.value
                self.log.info("Message from Producer : {}".format(value))
                value = json.loads(value)
                md5_value, md5_path = value["md5_value"], value["md5_path"]
                status = self.submit_2_cuckoo(file_path=md5_path)
                self.to_book_keeping_db(md5_value=md5_value, status=status, book_keeping_db=book_keeping_db)
            except Exception as e:
                self.log.error("Error : {}".format(e))

    def main(self):
        book_keeping_db = self.config["environment"]["mongo"]["book_keeping"]
        broker_list = self.config["environment"]["kafka"]["broker_list"]
        topic_name = self.config["environment"]["kafka"]["topic_name"]

        kafka = KafkaUtil(broker_list=broker_list)
        self.consume(book_keeping_db, kafka, topic_name)


if __name__ == '__main__':
    perform_analysis = PerformAnalysis()
    perform_analysis.main()
