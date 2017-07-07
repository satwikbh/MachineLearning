import os


class HelperFunction:
    def __init__(self):
        pass

    @staticmethod
    def get_files_with_extension(extension, path):
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
    def get_nearest_power_of_two(val):
        index = 0
        while val != 0:
            index += 1
            val = val >> 1
        return index
