from Utils.LoggerUtil import LoggerUtil
from kafka import KafkaConsumer, KafkaProducer


class KafkaUtil:
    # CHECKME : Initializing the producer and consumer in the init section.
    # Does this have any benefits or this type or invocation is error prone.
    # FIXME : Need to check the above.
    def __init__(self, broker_list):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.broker_list = broker_list
        self.producer = KafkaProducer(bootstrap_servers='localhost:1234')
        self.consumer = KafkaConsumer(bootstrap_servers='localhost:1234')

    def get_producer(self):
        return self.producer

    def get_consumer(self):
        return self.consumer
