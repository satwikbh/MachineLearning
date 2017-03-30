import ConfigParser
import logging
import time
import urllib

import gensim
from gensim import corpora
from pymongo import MongoClient

config = ConfigParser.RawConfigParser()
config.read('config.properties')

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
# ipython sometimes messes up the logging setup; restore
logging.root.level = logging.INFO

log = logging.getLoggerClass()


class TopicModelling:
    def __init__(self):
        self.doc2bow = dict()

    @staticmethod
    def get_client(address, port, username, password, auth_db):
        return MongoClient(
            "mongodb://" + username + ":" + urllib.quote(password) + "@" + address + ":" + port + "/" + auth_db)

    @staticmethod
    def get_bow_for_behavior_feature(feature, doc):
        bow = list()
        for key, value in doc.items():
            if isinstance(value, list):
                bow += list
            else:
                log.error("In feature {} \nSomething strange at this Key :{} \nValue : {}".format(feature, key, value))
        return bow

    def get_bow_for_network_feature(self, feature, doc):
        bow = list()
        for key, value in doc.items():
            if isinstance(value, dict):
                self.get_bow_for_network_feature(feature, value)
            elif isinstance(value, list):
                bow += value
            else:
                log.error("In feature {} \nSomething strange at this Key :{} \nValue : {}".format(feature, key, value))
        return bow

    @staticmethod
    def get_bow_for_statistic_feature(feature, doc):
        bow = list()
        if isinstance(doc, list):
            bow += doc
        else:
            log.error("Feature {} doesn't have {} type as value.".format(feature, type(doc)))

    @staticmethod
    def get_bow_for_static_feature(doc, feature):
        bow = list()
        for key, value in doc.items():
            if isinstance(value, list):
                bow += list
            if isinstance(value, dict):
                log.error("In feature {} \nSomething strange at this Key :{} \nValue : {}".format(feature, key, value))
        return bow

    def get_bow_for_each_document(self, document, feature):
        list_of_keys = document.values()[0].keys()
        if feature == "behavior" and feature in list_of_keys:
            behavior = document.values()[0].get(feature)
            return self.get_bow_for_behavior_feature(feature, behavior)
        elif feature == "network" and feature in list_of_keys:
            network = document.values()[0].get(feature)
            return self.get_bow_for_network_feature(feature, network)
        elif feature == "static" and feature in list_of_keys:
            static = document.values()[0].get(feature)
            return self.get_bow_for_static_feature(feature, static)
        elif feature == "statSignatures" and feature in list_of_keys:
            statistic = document.values()[0].get(feature)
            return self.get_bow_for_statistic_feature(feature, statistic)
        else:
            log.error("Feature other than behavior, network, static, statistic accessed.")
            return None

    def parse_each_document(self, cursor):
        for document in cursor:
            feature = document.get("feature")
            value = document.get("value")
            self.doc2bow[document.get("key")] = self.get_bow_for_each_document(value, feature)

    def lda_model(self):
        """
        Run the LDA model on the doc2bow matrix.
        :return:
        """
        # Creating the term dictionary of our courpus, where every unique term is assigned an index.
        values = self.doc2bow.values()
        dictionary = corpora.Dictionary(values)

        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in values]

        # Creating the object for LDA model using gensim library
        LDA = gensim.models.ldamodel.LdaModel

        # Running and Training LDA model on the document term matrix.
        ldamodel = LDA(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50)
        return ldamodel.print_topics(num_topics=3, num_words=3)

    def main(self):
        """
        The Main method
        """
        log.info("~~~~~~~ Program started ~~~~~~~")
        start_time = time.time()
        db_address = config.get("MongoDetails", "address")
        db_port = config.get("MongoDetails", "port")
        uname = config.get("MongoDetails", "username")
        password = urllib.quote(config.get("MongoDetails", "password"))
        auth_db = config.get("MongoDetails", "auth_db")
        db_name = config.get("MongoDetails", "db_name")
        collection_name = config.get("MongoDetails", "collection_name")

        client = self.get_client(db_address, db_port, uname, password, auth_db)
        db = client[db_name]
        collection = db[collection_name]

        query = {"key": {'$exists': True}}
        cursor = collection.find(query)
        self.parse_each_document(cursor)

        log.info("~~~~~~~ Printing the LDA model ~~~~~~~")
        log.info(self.lda_model())
        log.info("~~~~~~~ Total time taken : {} ~~~~~~~".format(time.time() - start_time))
