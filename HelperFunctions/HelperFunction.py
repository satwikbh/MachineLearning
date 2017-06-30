import os


class HelperFunction:
    def __init__(self):
        pass

    @staticmethod
    def is_file_present(fname):
        return os.path.isfile(fname)

    @staticmethod
    def get_nearest_power_of_two(val):
        index = 0
        while val != 0:
            index += 1
            val = val >> 1
        return index
