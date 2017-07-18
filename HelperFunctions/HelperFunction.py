import os
import math


class HelperFunction:
    def __init__(self):
        pass

    @staticmethod
    def get_threshold_point(list_of_nums, percentage):
        count = 0
        threshold = sum(list_of_nums) * percentage
        for x in xrange(len(list_of_nums)):
            if count > threshold:
                break
            count += list_of_nums[x]
        return x

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
    def is_file_present(fname):
        return os.path.isfile(fname)

    @staticmethod
    def create_dir_if_absent(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def get_full_path(directory, fname):
        return os.path.join(directory, fname)
