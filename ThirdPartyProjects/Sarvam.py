import numpy as np
import os
import array
import glob
import gist
import urllib

from time import time
from scipy.misc import imsave
from keras.preprocessing.image import img_to_array, load_img

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from Utils.DBUtils import DBUtils


class Sarvam:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.db_utils = DBUtils()

    def get_collection(self):
        username = self.config['environment']['mongo']['username']
        pwd = self.config['environment']['mongo']['password']
        password = urllib.quote(pwd)
        address = self.config['environment']['mongo']['address']
        port = self.config['environment']['mongo']['port']
        auth_db = self.config['environment']['mongo']['auth_db']
        is_auth_enabled = self.config['environment']['mongo']['is_auth_enabled']

        client = self.db_utils.get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=is_auth_enabled,
                                          username=username, password=password)

        db_name = self.config['environment']['mongo']['db_name']
        db = client[db_name]

        sarvam_coll_name = self.config['environment']['mongo']['sarvam_coll_name']
        sarvam_collection = db[sarvam_coll_name]

        return sarvam_collection

    @staticmethod
    def get_list_of_binaries(binaries_path):
        list_of_binaries = glob.glob(binaries_path + "/" + "VirusShare_*")
        return list_of_binaries

    @staticmethod
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

    def perform_sarvam(self, sarvam_collection, list_of_binaries, images_path):
        bulk = sarvam_collection.initialize_unordered_bulk_op()
        documents = dict()
        self.log.info("Total number of binaries : {}".format(len(list_of_binaries)))
        for index, binary in enumerate(list_of_binaries):
            if index % 1000 == 0:
                self.log.info("Working on iter : #{}".format(index / 1000))
            g = self.get_image(binary)
            imsave(images_path + "/" + str(binary) + ".png", g)
            image = load_img("/tmp/temp/7da61d59c52b3350f3b96926cf5fb94a215b1caa55cb69b2578aaa6e39afdfa3.png",
                             target_size=(64, 64, 3))
            arr = img_to_array(image).astype('uint8')

            gist_features = gist.extract(arr)
            feature = gist_features[0:320]
            documents["binary_name"] = binary
            documents["feature"] = feature.tolist()
            bulk.insert(documents)
        try:
            bulk.execute()
        except Exception as e:
            self.log.error("Error : {}".format(e))

    def main(self):
        start_time = time()
        binaries_path = self.config["sarvam"]["binaries_path"]
        images_path = self.config["sarvam"]["images_path"]
        sarvam_collection = self.get_collection()
        list_of_binaries = self.get_list_of_binaries(binaries_path)
        self.perform_sarvam(sarvam_collection, list_of_binaries, images_path)
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    sarvam = Sarvam()
    sarvam.main()
