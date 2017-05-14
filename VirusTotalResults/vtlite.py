#!/usr/bin/python
# Virus Total API Integration Script
# Built on VT Test Script from: Adam Meyers ~ CrowdStrike
# Rewirtten / Modified / Personalized: Chris Clark ~ GD Fidelis CyberSecurity
# If things are broken let me know chris@xenosec.org
# No Licence or warranty expressed or implied, use however you wish! 

import json, urllib, urllib2, argparse, hashlib, re, sys
from pprint import pprint


class vtAPI():
    def __init__(self):
        self.api = 'ad789e9c1d04904b4ca1a542d1eec0179f283c35180ac468cfb3d1fdb74caade'
        self.base = 'https://www.virustotal.com/vtapi/v2/'

    def getReport(self, md5):
        param = {'resource': md5, 'apikey': self.api}
        url = self.base + "file/report"
        data = urllib.urlencode(param)
        result = urllib2.urlopen(url, data)
        jdata = json.loads(result.read())
        return jdata

    def rescan(self, md5):
        param = {'resource': md5, 'apikey': self.api}
        url = self.base + "file/rescan"
        data = urllib.urlencode(param)
        result = urllib2.urlopen(url, data)
        print "\n\tVirus Total Rescan Initiated for -- " + md5 + " (Requery in 10 Mins)"


# Md5 Function

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
    if it['response_code'] == 0:
        print md5 + " -- Not Found in VT"
        return 0
    print "\n\tResults for MD5: ", it['md5'], "\n\n\tDetected by: ", it['positives'], '/', it['total'], '\n'
    if 'Sophos' in it['scans']:
        print '\tSophos Detection:', it['scans']['Sophos']['result'], '\n'
    if 'Kaspersky' in it['scans']:
        print '\tKaspersky Detection:', it['scans']['Kaspersky']['result'], '\n'
    if 'ESET-NOD32' in it['scans']:
        print '\tESET Detection:', it['scans']['ESET-NOD32']['result'], '\n'

    print '\tScanned on:', it['scan_date']

    if jsondump:
        if collection:
            collection.insert(it)
        else:
            jsondumpfile = open("VTDL" + md5 + ".json", "w")
            pprint(it, jsondumpfile)
            jsondumpfile.close()
            print "\n\tJSON Written to File -- " + "VTDL" + md5 + ".json"

    if verbose:
        print '\n\tVerbose VirusTotal Information Output:\n'
        for x in it['scans']:
            print '\t', x, '\t' if len(x) < 7 else '', '\t' if len(x) < 14 else '', '\t', it['scans'][x][
                'detected'], '\t', it['scans'][x]['result']


def main(collection, malware_md5):
    try:
        vt = vtAPI()
        md5 = checkMD5(malware_md5)
        parse(vt.getReport(md5), md5, False, True, collection)
        vt.rescan(md5)
    except Exception as e:
        print(e)
