from opencage.geocoder import OpenCageGeocode
from time import sleep
from random import randint

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil


class OpenCageGeoCoderApi:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.key = self.config["geo_coding"]["open_cage_key"]

    @staticmethod
    def get_county_state_country(result):
        county, state, country = [""] * 3
        if "county" in result:
            county = result["county"]
        if "state" in result:
            state = result["state"]
        if "country" in result:
            country = result["country"]
        return county, state, country

    def get_from_lat_long(self, key, latitude, longitude):
        """
        :param key:
        :param latitude:
        :param longitude:
        :return:
        """
        county, state, country, iso_2, iso_3 = [""] * 5
        try:
            geocoder = OpenCageGeocode(key)
            result_list = geocoder.reverse_geocode(latitude, longitude)
            sleep(randint(1, 3))
            for result in result_list:
                res = result["components"]
                county, state, country = self.get_county_state_country(result=res)
        except Exception as e:
            self.log.error("Error : {}".format(e))
        return ",".join([county, state, country]), iso_2, iso_3
