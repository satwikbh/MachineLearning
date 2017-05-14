import urllib
import logging
from collections import defaultdict
from pymongo import MongoClient
import falconn as falc
import numpy as np

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
# ipython sometimes messes up the logging setup; restore
logging.root.level = logging.INFO

log = logging.getLogger(__name__)

doc2bow = defaultdict(list)


def get_bow_for_behavior_feature(feature, doc):
    bow = list()
    for key, value in doc.items():
        if isinstance(value, list):
            bow += value
        else:
            log.error("In feature {} \nSomething strange at this Key :{} \nValue : {}".format(feature, key, value))
    return bow


def get_bow_for_network_feature(feature, doc):
    bow = list()
    for key, value in doc.items():
        if isinstance(value, dict):
            bow += get_bow_for_network_feature(feature, value)
        elif isinstance(value, list):
            bow += [str(s) for s in value if isinstance(s, int)]
        else:
            log.error("In feature {} \nSomething strange at this Key :{} \nValue : {}".format(feature, key, value))
    return bow


def get_bow_for_statistic_feature(feature, doc):
    bow = list()
    if isinstance(doc, list):
        bow += doc
    else:
        log.error("Feature {} doesn't have {} type as value.".format(feature, type(doc)))


def get_bow_for_static_feature(feature, doc):
    bow = list()
    for key, value in doc.items():
        if isinstance(value, list):
            bow += value
        if isinstance(value, dict):
            log.error("In feature {} \nSomething strange at this Key :{} \nValue : {}".format(feature, key, value))
    return bow


def get_bow_for_each_document(document, feature):
    if feature == "behavior":
        behavior = document.values()[0].get(feature)
        return get_bow_for_behavior_feature(feature, behavior)
    elif feature == "network":
        network = document.values()[0].get(feature)
        return get_bow_for_network_feature(feature, network)
    elif feature == "static":
        static = document.values()[0].get(feature)
        return get_bow_for_static_feature(feature, static)
    elif feature == "statSignatures":
        statistic = document.values()[0].get(feature)
        return get_bow_for_statistic_feature(feature, statistic)
    else:
        log.error("Feature other than behavior, network, static, statistic accessed.")
        return None


def parse_each_document(list_of_docs, collection):
    for each_document in list_of_docs:
        cursor = collection.find({"key": each_document})
        for each in cursor:
            feature = each.get("feature")
            value = each.get("value")
            if feature == "behavior" or feature == "network" or feature == "static" or feature == "statSignatures":
                list_of_keys = value.values()[0].keys()
                if feature in list_of_keys:
                    d2b = get_bow_for_each_document(value, feature)
                    if d2b is not None:
                        doc2bow[each.get("key")] += d2b


def convert2vec():
    """
    Generate & return the feature vector.
    :return: 
    """
    flat_list = [item for sublist in doc2bow.values() for item in sublist]
    cluster = list(set(flat_list))
    feature_vector = list(list())

    for each in doc2bow.values():
        temp = list()
        for here in cluster:
            if here in each:
                temp.append(1)
            else:
                temp.append(0)
        feature_vector.append(temp)
    return feature_vector


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
    lsh_params.l = 10

    # we build 20-bit hashes so that each table has
    # 2^20 bins; this is a good choise since 2^20 is of the same
    # order of magnitude as the number of data points
    # falc.compute_number_of_hash_functions(20, lsh_params)
    falc.compute_number_of_hash_functions(10, lsh_params)
    index = falc.LSHIndex(lsh_params)

    return index


def lsh():
    hcp = convert2vec()
    inp = np.array(hcp)

    # Converting the type as float32. This is what the falconn package expects.
    dataset = inp.astype(np.float32)

    # Centering the data
    dataset -= np.mean(dataset, axis=0)

    # Using default parameters for now. Need to change this as per out requirement.
    # params = falc.get_default_parameters(dimension=dataset.shape[1], num_points=dataset.shape[0])

    # Custom parameters for LSH
    index = set_lsh_parameters(dataset)
    index.setup(dataset=dataset)

    query = dataset[0]

    something = index.find_k_nearest_neighbors(k=10, query=query)


def main():
    username = "admin"
    password = "goodDevelopers@123"
    address = "localhost"
    port = "27017"
    auth_db = "admin"

    client = MongoClient(
        "mongodb://" + username + ":" + urllib.quote(password) + "@" + address + ":" + port + "/" + auth_db)

    db = client['cuckoo']
    coll = db['cluster2db']

    query = {"key": {'$exists': True}}
    list_of_docs = coll.find(query).distinct("key")

    parse_each_document(list_of_docs, coll)
    lsh()


if __name__ == '__main__':
    main()
