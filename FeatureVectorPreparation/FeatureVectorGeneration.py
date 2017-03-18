import ConfigParser
import csv
import logging
import os
import pickle as pi

from pymongo import MongoClient

config = ConfigParser.RawConfigParser()
config.read('Config.properties')

from collections import defaultdict

log = logging.getLogger(__name__)
LOG_FILENAME = config.get('Mappings', 'mappings.logfile')
LOG_LEVEL = config.get('Mappings', 'loglevel')

logging.basicConfig(filename=LOG_FILENAME, level=LOG_LEVEL)


class FeatureVectorGeneration():
    files_cluster = set()
    keys_cluster = set()
    summary_cluster = set()
    mutexes_cluster = set()
    executed_commands_cluster = set()
    domain_cluster = set()
    udp_cluster = set()
    hosts_cluster = set()
    dns_cluster = set()
    dirents_cluster = set()
    fn_name_cluster = set()
    dlls_cluster = set()
    sections_cluster = set()
    peid_signatures_cluster = set()

    fingerprint = defaultdict(list)

    cluster_path = ""
    cuckoo_root = ""

    @staticmethod
    def get_client():
        return MongoClient(
            "mongod://" + config.get("MongoProperties", "address") + ":" + config.get("MongoProperties", "port"))

    @staticmethod
    def get_collection(client):
        db = client.get_database(config.get("MongoProperties", "db_name"))
        collection = db.get_collection(config.get("MongoProperties", "collection_name"))
        return collection

    def load_props(self):
        behavior_keys = []
        network_keys = []
        static_keys = []
        file_list = os.listdir(self.cluster_path)
        for file in file_list:
            if file.endswith(".behaviorDump.cluster"):
                behavior_keys.append(file)
            if file.endswith(".networkDump.cluster"):
                network_keys.append(file)
            if file.endswith(".staticDump.cluster"):
                static_keys.append(file)
        return behavior_keys, network_keys, static_keys

    @staticmethod
    def make_fp(value, meta_list):
        meta_key_cluster = set()
        key_cluster = set()

        for each in value:
            for here in each.split("\\"):
                key_cluster.add(here.lower())

        fingerprint = []
        for each in meta_list:
            for here in each.split("\\"):
                meta_key_cluster.add(here.lower())

        for each in meta_key_cluster:
            if each in key_cluster:
                fingerprint.append("1")
            else:
                fingerprint.append("0")

        return fingerprint

    def generate_fingerprint(self, key, value):
        # TODO : Make this switch instead of if else
        if key == "files":
            # TODO : Implement Graph approach.
            return ''.join(self.make_fp(value, self.files_cluster))

        if key == "keys":
            return ''.join(self.make_fp(value, self.keys_cluster))

        if key == "summary":
            return ''.join(self.make_fp(value, self.summary_cluster))

        if key == "mutexes":
            return ''.join(self.make_fp(value, self.mutexes_cluster))

        if key == "executed_commands":
            return ''.join(self.make_fp(value, self.executed_commands_cluster))

        if key == "domains":
            return ''.join(self.make_fp(value, self.domain_cluster))

        if key == "udp":
            return ''.join(self.make_fp(value, self.udp_cluster))

        if key == "hosts":
            return ''.join(self.make_fp(value, self.hosts_cluster))

        if key == "dns":
            return ''.join(self.make_fp(value, self.dns_cluster))

        if key == "dirents":
            return ''.join(self.make_fp(value, self.dirents_cluster))

        if key == "fn_name":
            return ''.join(self.make_fp(value, self.fn_name_cluster))

        if key == "dlls":
            return ''.join(self.make_fp(value, self.dlls_cluster))

        if key == "sections":
            return ''.join(self.make_fp(value, self.sections_cluster))

        if key == "peid_signatures":
            return ''.join(self.make_fp(value, self.peid_signatures_cluster))

    def meta_cluster(self, key, value):

        if key == "files":
            for here in value:
                self.files_cluster.add(here.lower())

        if key == "keys":
            for here in value:
                self.keys_cluster.add(here.lower())

        if key == "summary":
            for here in value:
                self.summary_cluster.add(here.lower())

        if key == "mutexes":
            for here in value:
                self.mutexes_cluster.add(here.lower())

        if key == "executed_commands":
            for here in value:
                self.executed_commands_cluster.add(here.lower())

        if key == "domains":
            for here in value:
                self.domain_cluster.add(here.lower())

        if key == "udp":
            for here in value:
                self.udp_cluster.add(str(here).lower())

        if key == "hosts":
            for here in value:
                self.hosts_cluster.add(here.lower())

        if key == "dns":
            for here in value:
                self.dns_cluster.add(here.lower())

        if key == "dirents":
            for here in value:
                self.dirents_cluster.add(here.lower())

        if key == "fn_name":
            for here in value:
                self.fn_name_cluster.add(here.lower())

        if key == "dlls":
            for here in value:
                self.dlls_cluster.add(here.lower())

        if key == "sections":
            for here in value:
                self.sections_cluster.add(here.lower())

        if key == "peid_signatures":
            for here in value:
                self.peid_signatures_cluster.add(here.lower())

    def behavior(self):
        for each_variant in self.behavior_dump.keys():
            val = self.behavior_dump.get(each_variant).keys()[0]
            for each_key in self.behavior_dump.get(each_variant).get(val).keys():
                key = each_key
                value = self.behavior_dump.get(each_variant).get(val).get(each_key)
                self.meta_cluster(key, value)

    def network(self):
        for each_variant in self.network_dump.keys():
            val = self.network_dump.get(each_variant).keys()[0]
            for each_key in self.network_dump.get(each_variant).get(val).keys():
                key = each_key
                value = self.network_dump.get(each_variant).get(val).get(each_key)
                self.meta_cluster(key, value)

    def static(self):
        for each_variant in self.static_dump.keys():
            val = self.static_dump.get(each_variant).keys()[0]
            for each_key in self.static_dump.get(each_variant).get(val).keys():
                key = each_key
                value = self.static_dump.get(each_variant).get(val).get(each_key)
                self.meta_cluster(key, value)

    def make_static(self):
        self.files_cluster = list(self.files_cluster)
        self.keys_cluster = list(self.keys_cluster)
        self.summary_cluster = list(self.summary_cluster)
        self.mutexes_cluster = list(self.mutexes_cluster)
        self.executed_commands_cluster = list(self.executed_commands_cluster)
        self.domain_cluster = list(self.domain_cluster)
        self.udp_cluster = list(self.udp_cluster)
        self.hosts_cluster = list(self.hosts_cluster)
        self.dns_cluster = list(self.dns_cluster)
        self.dirents_cluster = list(self.dirents_cluster)
        self.fn_name_cluster = list(self.fn_name_cluster)
        self.dlls_cluster = list(self.dlls_cluster)
        self.sections_cluster = list(self.sections_cluster)
        self.peid_signatures_cluster = list(self.peid_signatures_cluster)

    def behavior_fp(self):
        fp = defaultdict(dict)
        for each_variant in self.behavior_dump:
            val = self.behavior_dump.get(each_variant).keys()[0]
            for each_key in self.behavior_dump.get(each_variant).get(val):
                key = each_key
                value = self.behavior_dump.get(each_variant).get(val).get(each_key)
                fp[key] = self.generate_fingerprint(key, value)
            self.fingerprint[each_variant].append(fp)

    def network_fp(self):
        fp = defaultdict(dict)
        for each_variant in self.network_dump:
            val = self.network_dump.get(each_variant).keys()[0]
            for each_key in self.network_dump.get(each_variant).get(val):
                key = each_key
                value = self.network_dump.get(each_variant).get(val).get(each_key)
                fp[key] = self.generate_fingerprint(key, value)
            self.fingerprint[each_variant].append(fp)

    def static_fp(self):
        fp = defaultdict(dict)
        for each_variant in self.static_dump:
            val = self.static_dump.get(each_variant).keys()[0]
            for each_key in self.static_dump.get(each_variant).get(val):
                key = each_key
                value = self.static_dump.get(each_variant).get(val).get(each_key)
                fp[key] = self.generate_fingerprint(key, value)
            self.fingerprint[each_variant].append(fp)

    @staticmethod
    def get_cluster_path():
        """
        Gets the current path and stores the dumps in cluster
        :return:
        """
        _current_dir = os.path.abspath(os.path.dirname("__file__"))
        # CUCKOO_ROOT = "/".join(_current_dir.split("/")[:-2])
        # path = CUCKOO_ROOT + "/cluster/"
        path = _current_dir + "/cluster/"
        log.info("Cluster Path: {0}".format(path))
        return path, _current_dir

    def get_path(self):
        path = self.cuckoo_root + "/fingerprint/"
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except Exception as e:
            log.error(e)
        return path

    def create_cluster_dumps(self, collection):
        """
        This method will create all the dumps.
        This is a one time process and once the dumps are created, the next iteration will use these dumps to prepare the next set of dumps.
        :return:
        """
        pi.dump(self.behavior_dump, open("Meta_Behavior.dump", "w"))
        pi.dump(self.network_dump, open("Meta_Network.dump", "w"))
        pi.dump(self.static_dump, open("Meta_Static.dump", "w"))

    @staticmethod
    def load_dump(cluster_path, dump_name):
        return pi.load(open(cluster_path + dump_name))

    def main(self):
        self.cluster_path, self.cuckoo_root = self.get_cluster_path()

        collection = self.get_collection(self.get_client())

        behavior_keys, network_keys, static_keys = self.load_props()

        for each in behavior_keys:
            self.behavior_dump = self.load_dump(self.cluster_path, each)
            self.behavior()

        for each in network_keys:
            self.network_dump = self.load_dump(self.cluster_path, each)
            self.network()

        for each in static_keys:
            self.static_dump = self.load_dump(self.cluster_path, each)
            self.static()

        self.make_static()

        self.create_cluster_dumps(collection)

        x = 0
        for each in behavior_keys:
            x += 1
            self.behavior_dump = self.load_dump(self.cluster_path, each)
            self.behavior_fp()
            print(x)

        for each in network_keys:
            self.network_dump = self.load_dump(self.cluster_path, each)
            self.network_fp()

        for each in static_keys:
            self.static_dump = self.load_dump(self.cluster_path, each)
            self.static_fp()

        with open(self.get_path() + 'mycsvfile.csv', 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in self.fingerprint.items():
                ans = ""
                for each in value:
                    ans += ''.join(each.values())
                writer.writerow([key, ans])

        f1 = open(self.get_path() + "fingerprint.fp", "wa")
        pi.dump(self.fingerprint, f1)
        f1.close()


if __name__ == "__main__":
    fvg = FeatureVectorGeneration()
    fvg.main()
