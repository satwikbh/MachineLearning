#!/usr/bin/python
# Virus Total API Integration Script
# Built on VT Test Script from: Adam Meyers ~ CrowdStrike
# Rewirtten / Modified / Personalized: Chris Clark ~ GD Fidelis CyberSecurity
# If things are broken let me know chris@xenosec.org
# No Licence or warranty expressed or implied, use however you wish! 

import hashlib
import json
import re
import urllib
import urllib2
from pprint import pprint


class vtAPI():
    def __init__(self):
        self.api = 'ad789e9c1d04904b4ca1a542d1eec0179f283c35180ac468cfb3d1fdb74caade'
        self.api_new = "f79f5999550593986e44b4df25d65e719db681cff42c952271aaaea912b2c852"
        self.base = 'https://www.virustotal.com/vtapi/v2/'

    def getReport(self, md5, switch):
        if switch:
            param = {'resource': md5, 'apikey': self.api}
        else:
            param = {'resource': md5, 'apikey': self.api_new}
        url = self.base + "file/report"
        data = urllib.urlencode(param)
        result = urllib2.urlopen(url, data).read()
        jdata = json.loads(result)
        return jdata


def checkMD5(checkval):
    if re.match(r"([a-fA-F\d]{32})", checkval) == None:
        md5 = md5sum(checkval)
        return md5.upper()
    else:
        return checkval.upper()


def md5sum(filename):
    fh = open(filename, 'rb')
    m = hashlib.md5()
    while True:
        data = fh.read(8192)
        if not data:
            break
        m.update(data)
    return m.hexdigest()


def parse(it, md5, verbose, jsondump, collection):
    if jsondump:
        if collection:
            it_dict = dict(it)
            it_dict['malware_source'] = "VirusShare_" + str(md5.lower())
            collection.insert(json.loads(json.dumps(it_dict)))
        else:
            jsondumpfile = open("VTDL" + md5 + ".json", "w")
            pprint(it, jsondumpfile)
            jsondumpfile.close()
            print "\n\tJSON Written to File -- " + "VTDL" + md5 + ".json"


def main(collection, malware_md5, switch):
    try:
        vt = vtAPI()
        md5 = checkMD5(malware_md5)
        parse(vt.getReport(md5, switch), md5, False, True, collection)
    except Exception as e:
        print(e, malware_md5)
