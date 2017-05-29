from pymongo import MongoClient


class DBUtils:
    def __init__(self):
        pass

    @staticmethod
    def get_client(address, port, username, password, auth_db, is_auth_enabled):
        if is_auth_enabled:
            client = MongoClient("mongodb://" + username + ":" + password + "@" + address + ":" + port + "/" + auth_db)
        else:
            client = MongoClient("mongodb://" + address + ":" + port + "/" + auth_db)
        return client
