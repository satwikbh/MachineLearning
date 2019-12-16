from requests import get
from time import sleep
from random import randint

from GeoCoding.OpenCageGeoCoderApi import OpenCageGeoCoderApi

from Utils.LoggerUtil import LoggerUtil
from Utils.DBUtils import DBUtils
from Utils.ConfigUtil import ConfigUtil


class IPDataApi:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.db_utils = DBUtils()
        self.config = ConfigUtil.get_config_instance()
        self.open_cage_api = OpenCageGeoCoderApi()
        self.key = self.config["geo_coding"]["open_cage_key"]
        self.url = self.config["geo_coding"]["ipapi_url"]

    def make_url(self, ip_addr):
        """
        url = "https://ipapi.co/" + str(ip_addr) + "/" + "json"
        :param ip_addr:
        :return:
        """
        url_string = self.url + str(ip_addr) + "/json/"
        return url_string

    @staticmethod
    def make_ip_coll_doc(latitude, longitude, location_string, iso_2, iso_3):
        return {
            "latitude": latitude,
            "longitude": longitude,
            "location_string": location_string,
            "iso_2": iso_2,
            "iso_3": iso_3
        }

    @staticmethod
    def make_lat_long_doc(ip_addr, location_string, iso_2, iso_3):
        return {
            "ip_addr": ip_addr,
            "location_string": location_string,
            "iso_2": iso_2,
            "iso_3": iso_3
        }

    def get_ip_data(self, ip_addr_list):
        ip_dict = dict()
        lat_long_dict = dict()
        for ip_addr in ip_addr_list:
            try:
                url = self.make_url(ip_addr=ip_addr)
                sleep(randint(1, 3))
                response = get(url).json()
                if "latitude" in response:
                    latitude = response["latitude"]
                else:
                    latitude = ""
                if "longitude" in response:
                    longitude = response["longitude"]
                else:
                    longitude = ""
                location_string, iso_2, iso_3 = self.open_cage_api.get_from_lat_long(key=self.key,
                                                                                     latitude=latitude,
                                                                                     longitude=longitude)
                ip_dict[str(ip_addr)] = self.make_ip_coll_doc(latitude, longitude, location_string, iso_2, iso_3)
                key = str(latitude) + "_" + str(longitude)
                lat_long_dict[key] = self.make_lat_long_doc(ip_addr, location_string, iso_2, iso_3)
            except Exception as e:
                self.log.error("Error : {}".format(e))
        return ip_dict, lat_long_dict
