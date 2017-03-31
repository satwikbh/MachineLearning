import ConfigParser
import logging
import time
import urllib
from collections import defaultdict

import gensim
from gensim import corpora
from pymongo import MongoClient

config = ConfigParser.RawConfigParser()
config.read('../Config.properties')

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
# ipython sometimes messes up the logging setup; restore
logging.root.level = logging.INFO

log = logging.getLogger(__name__)


class TopicModelling:
    def __init__(self):
        self.doc2bow = defaultdict(list)

    @staticmethod
    def get_client_wo_auth(address, port):
        return MongoClient(
            "mongodb://" + address + ":" + port)

    @staticmethod
    def get_client(address, port, username, password, auth_db):
        return MongoClient(
            "mongodb://" + username + ":" + urllib.quote(password) + "@" + address + ":" + port + "/" + auth_db)

    @staticmethod
    def get_bow_for_behavior_feature(feature, doc):
        bow = list()
        for key, value in doc.items():
            if isinstance(value, list):
                bow += value
            else:
                log.error("In feature {} \nSomething strange at this Key :{} \nValue : {}".format(feature, key, value))
        return bow

    def get_bow_for_network_feature(self, feature, doc):
        bow = list()
        for key, value in doc.items():
            if isinstance(value, dict):
                bow += self.get_bow_for_network_feature(feature, value)
            elif isinstance(value, list):
                bow += [str(s) for s in value if isinstance(s, int)]
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
    def get_bow_for_static_feature(feature, doc):
        bow = list()
        for key, value in doc.items():
            if isinstance(value, list):
                bow += value
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
        for each_document in cursor:
            feature = each_document.get("feature")
            value = each_document.get("value")
            d2b = self.get_bow_for_each_document(value, feature)
            if d2b is not None:
                self.doc2bow[each_document.get("key")] += d2b

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
        ldamodel = LDA(doc_term_matrix, num_topics=100, id2word=dictionary, passes=500)

        # Save the model.
        LDA.save(ldamodel, open("LDA.model", "w"))

        # log.info(ldamodel.print_topics(num_topics=3, num_words=3))
        log.info(ldamodel.show_topics())

    def main(self):
        """
        The Main method
        """
        log.info("~~~~~~~ Program started ~~~~~~~")
        start_time = time.time()
        is_auth_enabled = config.get("MongoProperties", "isAuthEnabled")
        db_address = config.get("MongoProperties", "address")
        db_port = config.get("MongoProperties", "port")
        uname = config.get("MongoProperties", "username")
        password = urllib.quote(config.get("MongoProperties", "password"))
        auth_db = config.get("MongoProperties", "auth_db")
        db_name = config.get("MongoProperties", "db_name")
        collection_name = config.get("MongoProperties", "fingerprint_collection")

        if is_auth_enabled:
            client = self.get_client_wo_auth(db_address, db_port)
        else:
            client = self.get_client(db_address, db_port, uname, password, auth_db)
        db = client[db_name]
        collection = db[collection_name]

        query = {"key": {'$exists': True}}
        cursor = collection.find(query).limit(100)
        self.parse_each_document(cursor)

        log.info("~~~~~~~ Printing the LDA model ~~~~~~~")
        log.info(self.lda_model())
        log.info("~~~~~~~ Total time taken : {} ~~~~~~~".format(time.time() - start_time))


if __name__ == '__main__':
    tm = TopicModelling()
    tm.main()
