import logging
import numpy as np
import pickle as pi
import urllib

from time import time
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.neighbors import BallTree
from multiprocessing import Pool, get_logger

from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils
from HelperFunctions.HelperFunction import HelperFunction
from PrepareData.LoadData import LoadData

config = ConfigUtil().get_config_instance()
helper = HelperFunction()
load_data = LoadData()
db_utils = DBUtils()
meta_dict = defaultdict()


def create_logger():
    logger = get_logger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("logs/multi_process.log")
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_collection():
    username = config['environment']['mongo']['username']
    pwd = config['environment']['mongo']['password']
    password = urllib.quote(pwd)
    address = config['environment']['mongo']['address']
    port = config['environment']['mongo']['port']
    auth_db = config['environment']['mongo']['auth_db']
    is_auth_enabled = config['environment']['mongo']['is_auth_enabled']

    client = db_utils.get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=is_auth_enabled,
                                 username=username, password=password)

    db_name = config['environment']['mongo']['db_name']
    db = client[db_name]

    sarvam_coll_name = config['environment']['mongo']['sarvam_coll_name']
    sarvam_collection = db[sarvam_coll_name]

    avclass_collection_name = config['environment']['mongo']['avclass_collection_name']
    avclass_collection = db[avclass_collection_name]

    return sarvam_collection, avclass_collection


def get_freq_dict(document):
    verbose = document["avclass"]["verbose"]
    md5 = document["md5"]
    if len(verbose) == 1:
        family_name, score = verbose[0]
        meta_dict[md5] = [{"family_name": family_name, "score": 1}]
    else:
        tmp = list()
        total_score = 0
        for inner_list in verbose:
            family_name, score = inner_list
            total_score += score
        for inner_list in verbose:
            family_name, score = inner_list
            tmp.append({"family_name": family_name, "score": (score * 1.0) / total_score})
        meta_dict[md5] = tmp


def get_avclass_dist(list_of_keys, avclass_collection):
    count = 0
    index = 0
    chunk_size = 1000

    while count < len(list_of_keys):
        log.info("Working on Iter : #{}".format(index))
        if count + chunk_size < len(list_of_keys):
            p_keys = list_of_keys[count: count + chunk_size]
        else:
            p_keys = list_of_keys[count:]

        query = [
            {"$match": {"md5": {"$in": p_keys}}},
            {"$project": {"avclass.verbose": 1, "md5": 1}},
            {"$addFields": {"__order": {"$indexOfArray": [p_keys, "$md5"]}}},
            {"$sort": {"__order": 1}}
        ]
        cursor = avclass_collection.aggregate(query)
        for doc in cursor:
            get_freq_dict(doc)
        count += chunk_size
        index += 1


def get_list_of_binaries(sarvam_collection):
    list_of_binaries = list()
    cursor = sarvam_collection.aggregate([{"$group": {"_id": '$binary'}}])
    for _ in cursor:
        list_of_binaries.append(_["_id"])
    return list_of_binaries


def sarvam_binary_predictions(list_of_binaries, sarvam_collection):
    binary_predictions = defaultdict()
    count = 0
    index = 0
    chunk_size = 1000

    while count < len(list_of_binaries):
        log.info("Working on Iter : #{}".format(index))
        if count + chunk_size < len(list_of_binaries):
            p_keys = list_of_binaries[count: count + chunk_size]
        else:
            p_keys = list_of_binaries[count:]

        query = [
            {"$match": {"binary": {"$in": p_keys}}},
            {"$project": {"feature": 1, "binary": 1}},
            {"$addFields": {"__order": {"$indexOfArray": [p_keys, "$md5"]}}},
            {"$sort": {"__order": 1}}
        ]
        cursor = sarvam_collection.aggregate(query)
        for doc in cursor:
            binary = doc["binary"]
            feature = doc["feature"]
            binary_predictions[binary] = feature
        count += chunk_size
        index += 1
    return binary_predictions


def compute_acc(binary_family, binary_values):
    tp = defaultdict()
    for x in binary_family:
        tp[x["family_name"]] = 0
        for y in binary_values:
            if x["family_name"] in [_["family_name"] for _ in y]:
                tp[x["family_name"]] = 1
    return max(tp.values())


def go_parallel(**kwargs):
    num_proc = kwargs["num_proc"]
    ball_tree_model_path = kwargs["ball_tree_model_path"]
    binary_predictions = kwargs["binary_predictions"]
    top_k = kwargs["top_k"]

    pool = Pool(processes=num_proc)
    params = [ball_tree_model_path, binary_predictions, top_k]
    p_place_holder = pool.map_async(evaluate_sarvam, params)
    result = p_place_holder.get()
    pool.close()
    pool.join()
    return result


def evaluate_sarvam(**kwargs):
    ball_tree_model_path = kwargs["ball_tree_model_path"]
    binary_predictions = kwargs["binary_predictions"]
    top_k = kwargs["top_k"]

    meta_acc = list()
    failed = list()
    ball_tree_model = joblib.load(ball_tree_model_path + "/" + "bt_model.pkl")
    binary_index = defaultdict()
    binary_index.default_factory = binary_index.__len__()
    for index, value in enumerate(binary_predictions.keys()):
        binary_index[index] = value

    log.info("Total number of binaries : {}".format(len(binary_predictions)))
    for binary, feature in binary_predictions.items():
        if binary in meta_dict:
            dist, ind = ball_tree_model.query([feature], k=top_k)
            binary_family = meta_dict[binary.split("VirusShare_")[1]]
            binary_values = list()
            for _ in ind[0]:
                binary_values.append(meta_dict[binary_index[_].split("VirusShare_")[1]])
            num = compute_acc(binary_family, binary_values)
            meta_acc.append(num)
        else:
            failed.append(binary)
    log.info("Binaries not found in dataset : {}".format(len(failed)))
    pi.dump(failed, open("failed.pkl", "w"))
    log.info("Accuracy at top k : {} is : {}".format(top_k, np.mean(meta_acc)))
    return meta_acc


def create_model(final_corpus, ball_tree_model_path):
    log.info("Creating Ball Tree for Corpus")
    corpus = np.asarray([np.asarray(document["feature"]) for document in final_corpus])
    ball_tree = BallTree(corpus)
    log.info("Saving Ball Tree model at the following path : {}".format(ball_tree_model_path))
    joblib.dump(ball_tree, ball_tree_model_path + "/" + "bt_model.pkl")
    return ball_tree


def main():
    start_time = time()
    ball_tree_model_path = config["sarvam"]["bt_model_path"]
    sarvam_collection, avclass_collection = get_collection()
    list_of_binaries = get_list_of_binaries(sarvam_collection)
    list_of_keys = helper.convert_from_vs_keys(list_of_vs_keys=list_of_binaries)
    get_avclass_dist(list_of_keys=list_of_keys, avclass_collection=avclass_collection)
    binary_predictions = sarvam_binary_predictions(list_of_binaries, sarvam_collection)
    go_parallel(num_proc=30, ball_tree_model_path=ball_tree_model_path, binary_predictions=binary_predictions,
                top_k=5)
    create_model(ball_tree_model_path=ball_tree_model_path, final_corpus=binary_predictions.values())
    log.info("Total time taken : {}".format(time() - start_time))


log = create_logger()
main()
