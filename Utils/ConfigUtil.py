import json


class ConfigUtil:
    def __init__(self):
        pass

    @staticmethod
    def get_config_instance():
        with open('config.json') as json_data_file:
            data = json.load(json_data_file)

        return data