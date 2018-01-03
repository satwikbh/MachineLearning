import math

import hickle as hkl
import numpy as np
import os
from scipy.sparse import vstack, load_npz
from sklearn.metrics import mean_squared_error


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

    @staticmethod
    def check_if_already_scanned(md5_value_list, c2db_collection):
        status_dict = dict()
        for md5_value in md5_value_list:
            key = "VirusShare_" + str(md5_value)
            cursor = c2db_collection.find({"key": key})
            if cursor.count() > 0:
                status_dict[md5_value] = True
            else:
                status_dict[md5_value] = False
        return status_dict
