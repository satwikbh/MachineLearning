import numpy as np
import os
import array
import glob
import gist
import logging

from skimage.io import imsave
from urllib.parse import quote
from keras.preprocessing.image import img_to_array, load_img
from multiprocessing import Pool, get_logger
from time import time

from HelperFunctions.HelperFunction import HelperFunction
from PrepareData.LoadData import LoadData
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils

config = ConfigUtil().get_config_instance()
helper = HelperFunction()
load_data = LoadData()
db_utils = DBUtils()

"""
Parallel Sarvam.
https://github.com/tuttieee/lear-gist-python
This version of gist needs to be installed by following the above instructions but not 
regular pip install.
"""


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
    password = quote(pwd)
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

    return sarvam_collection


def get_list_of_binaries(binaries_path):
    list_of_binaries = glob.glob(binaries_path + "/" + "VirusShare_*")
    return list_of_binaries


def get_image(binary):
    f = open(binary, 'rb')
    ln = os.path.getsize(binary)
    width = 256
    rem = ln % width
    a = array.array("B")
    a.fromfile(f, ln - rem)
    f.close()
    g = np.reshape(a, (len(a) / width, width))
    g = np.uint8(g)
    return g


def perform_sarvam(**kwargs):
    list_of_binaries = kwargs["list_of_binaries"]
    images_path = kwargs["images_path"]
    bulk = kwargs["bulk"]

    log.info("Total number of binaries : {}".format(len(list_of_binaries)))
    for index, binary in enumerate(list_of_binaries):
        try:
            documents = dict()
            bin_name = binary.split("/")[-1]
            if index % 1000 == 0:
                log.info("Working on iter : #{}".format(index / 1000))
            g = get_image(binary)
            image_name = images_path + "/" + str(bin_name) + ".png"
            imsave(image_name, g)
            image = load_img(image_name, target_size=(64, 64, 3))
            arr = img_to_array(image).astype('uint8')
            gist_features = gist.extract(arr)
            feature = gist_features[0:320]
            documents["binary"] = bin_name
            documents["feature"] = feature.tolist()
            bulk.insert(documents)
        except Exception as e:
            log.error("Error at binary : {}\nError is : {}".format(binary, e))


def go_parallel(**kwargs):
    num_proc = kwargs["num_proc"]
    sarvam_collection = kwargs["sarvam_collection"]
    list_of_binaries = kwargs["list_of_binaries"]
    images_path = kwargs["images_path"]

    bulk = sarvam_collection.initialize_unordered_bulk_op()
    log.info("Number of threads used are : {}".format(num_proc))
    pool = Pool(processes=num_proc)
    params = [list_of_binaries, images_path, bulk]
    p_place_holder = pool.map_async(perform_sarvam, params)
    p_place_holder.get()
    pool.close()
    pool.join()
    try:
        bulk.execute()
    except Exception as e:
        log.error("Error at bulk execute : {}".format(e))


def main():
    start_time = time()
    binaries_path = config["sarvam"]["binaries_path"]
    images_path = config["sarvam"]["images_path"]
    num_proc = 30
    sarvam_collection = get_collection()
    list_of_binaries = get_list_of_binaries(binaries_path)
    go_parallel(num_proc=num_proc, sarvam_collection=sarvam_collection, list_of_binaries=list_of_binaries,
                images_path=images_path)
    perform_sarvam(sarvam_collection=sarvam_collection, list_of_binaries=list_of_binaries, images_path=images_path)
    log.info("Total time taken : {}".format(time() - start_time))


log = create_logger()
main()
