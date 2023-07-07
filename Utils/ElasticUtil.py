from elasticsearch import Elasticsearch

from LoggerUtil import LoggerUtil
from ConfigUtil import ConfigUtil


class ElasticUtil:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()

    def get_es_client(self, host, port):
        try:
            es_client = Elasticsearch([
                {'host': host, 'port': port}
            ])
            return es_client
        except Exception as e:
            self.log.error("Error : {}".format(e))
