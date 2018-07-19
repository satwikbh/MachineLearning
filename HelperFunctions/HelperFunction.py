import matplotlib

matplotlib.use('Agg')

import math
import hickle as hkl
import numpy as np
import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from scipy.sparse import vstack, load_npz
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class HelperFunction:
    def __init__(self):
        pass

    @staticmethod
    def cursor_to_list(cursor, identifier):
        list_of_keys = list()
        for each in cursor:
            list_of_keys.append(each[identifier])
        return list_of_keys

    @staticmethod
    def convert_to_vs_keys(list_of_keys):
        """
        Convert a list of md5 keys to VirusShare_ format.
        :param list_of_keys:
        :return:
        """
        new_list_of_keys = list()
        for each_key in list_of_keys:
            new_list_of_keys.append("VirusShare_" + each_key)
        return new_list_of_keys

    @staticmethod
    def convert_from_vs_keys(list_of_vs_keys):
        new_list_of_keys = list()
        for each_key in list_of_vs_keys:
            new_list_of_keys.append(each_key.split("_")[1])
        return new_list_of_keys

    @staticmethod
    def frange(start, stop, step):
        x = start
        while x < stop:
            yield x
            x += step

    @staticmethod
    def stack_matrix(list_of_matrices):
        return vstack(list_of_matrices)

    @staticmethod
    def get_threshold_point(list_of_nums, percentage):
        count = 0
        threshold = sum(list_of_nums) * percentage
        for index, value in enumerate(list_of_nums):
            if count > threshold:
                break
            count += value
        return index

    @staticmethod
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    @staticmethod
    def nearest_power_of_two(shape):
        value = 1 << (shape - 1).bit_length()
        return int(math.log(value, 2))

    @staticmethod
    def get_files_starts_with_extension(extension, path):
        all_files = list()
        for each_file in os.listdir(path):
            if each_file.startswith(extension):
                all_files.append(os.path.join(path, each_file))
        return all_files

    @staticmethod
    def get_files_ends_with_extension(extension, path):
        all_files = list()
        for each_file in os.listdir(path):
            if each_file.endswith(extension):
                all_files.append(os.path.join(path, each_file))
        return all_files

    @staticmethod
    def is_file_present(fname):
        return os.path.isfile(fname)

    @staticmethod
    def create_dir_if_absent(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def get_full_path(directory, fname):
        return os.path.join(directory, fname)

    @staticmethod
    def flatten_list(nested_list):
        """
        Flatten a list of lists
        :param nested_list:
        :return:
        """
        flat_list = [item for sublist in nested_list for item in sublist]
        return flat_list

    @staticmethod
    def open_np_files(list_of_files):
        matrix = list()
        for each in list_of_files:
            fv = load_npz(each)
            matrix.append(fv)
        return matrix

    @staticmethod
    def open_files(list_of_files):
        matrix = list()
        for each in list_of_files:
            fv = hkl.load(each)
            matrix.append(fv)
        return matrix

    @staticmethod
    def mean_square_error(source, target):
        mse = mean_squared_error(source, target)
        return mse

    @staticmethod
    def is_nested_list(l):
        return_list = list()
        for each in l:
            if isinstance(each, list):
                return_list += each
            else:
                return_list.append(each)
        return return_list

    @staticmethod
    def centroid_np(arr):
        """
        Find the centroid of a array of points.
        :param arr:
        :return:
        """
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return sum_x / length, sum_y / length

    def check_if_already_scanned(self, md5_value_list, c2db_collection, chunk=1000):
        """
        In case a malware is scanned, it's md5 value will be present in the database.
        Will check if the malware was scanned by checking the md5 in db.
        :param md5_value_list: list of md5 of the malware executables.
        :param c2db_collection: the c2db collection
        :param chunk: chunk size for the in query. If its large, then mongo will throw >16mb error.
        :return:
        """
        status_dict = dict()
        md5_value_list = self.convert_to_vs_keys(md5_value_list)
        count = 0
        while count < len(md5_value_list):
            if count + count > len(md5_value_list):
                temp_list = md5_value_list[count:]
            else:
                temp_list = md5_value_list[count: count + chunk]
            cursor = c2db_collection.find({"key": {"$in": temp_list}}, {"key": 1})
            for doc in cursor:
                if "key" in doc:
                    status_dict[doc["key"]] = True
                else:
                    status_dict[doc["key"]] = False
            count += chunk
        return status_dict

    @staticmethod
    def plot_cnf_matrix(cnf_matrix):
        # TODO : Move to Plotting.py
        df_cm = pd.DataFrame(cnf_matrix, index=[i for i in range(cnf_matrix.shape[0])],
                             columns=[i for i in range(cnf_matrix.shape[0])])
        plt.figure(figsize=(20, 10))
        sn.heatmap(df_cm, annot=True)
        return plt

    @staticmethod
    def validation_split(input_matrix, labels, test_size):
        x_train, x_test, y_train, y_test = train_test_split(input_matrix, labels, test_size=test_size, random_state=0,
                                                            stratify=labels)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def make_unicode(input_str):
        input_str = str(input_str)
        if type(input_str) != unicode:
            input_str = input_str.decode('utf-8')
            return input_str
        else:
            return input_str
