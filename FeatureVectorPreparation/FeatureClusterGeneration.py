import pickle as pi
import os

from collections import defaultdict
from Utils.LoggerUtil import LoggerUtil


class FeatureClusterGeneration:
    """
    This class generates the Fingerprint of a Malware
    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    name = ""
    cluster_path = ""

    cluster_list = []
    matches = []
    fingerprint = []
    abstract_fingerprint = []
    list_feature_vectors = []

    failed_analyses = []

    stats_dump = defaultdict(dict)  # Is a dict of dict
    signature_dump = defaultdict(dict)  # Is a dict of dict
    malheur_dump = defaultdict(dict)  # Is a dict of dict
    static_dump = defaultdict(dict)  # Is a dict of dict
    network_dump = defaultdict(dict)  # Is a dict of dict
    behavior_dump = defaultdict(dict)  # Is a dict of dict
    unknown_features = defaultdict(dict)  # Is a dict of dict

    @staticmethod
    def is_cluster_present(name):
        """
        Checks if the cluster is present or not.
        :param name:
        :return:
        """
        return os.path.isfile(name + ".cluster")

    def list_of_failed_analyses(self, error_list, path):
        """
        This method checks for the failed analyses and if any then it will add them to the list.
        :param error_list:
        :param path:
        :return:
        """
        if len(error_list) > 0:
            self.failed_analyses.append(path)
            return True
        return False

    def alert_unknown_features(self, md5, name):
        """
        This method will be invoked when features of certain malware are activated.
        Example we don't know what procmemory does and also for test variants the output is empty.
        Hence, we can't add it to feature list.
        If in future for any malware if it is activated then we will take care of it.
        :param md5:
        :param name:
        :return:
        """
        self.unknown_features[md5].append(name)

    def get_name(self, results):
        """
        Gets the name of the file being analyzed.
        :param results:
        :return:
        """
        if "target" in results and results.get("target") is not None:
            target = results.get("target")
            if "file" in target and target.get("file") is not None:
                file_info = target.get("file")
                if "name" in file_info and file_info.get("name") is not None:
                    self.name = file_info.get("name")

    def debug(self, md5, results):
        """
        The debug results which collect the failed analyzes.
        :param results:
        :return:
        """

        if "debug" in results and results.get("debug") is not None:
            debug = results.get("debug")
            if "errors" in debug and self.list_of_failed_analyses(debug.get("errors"), self.name):
                return True
            else:
                return False

    def peid_signatures(self, pe_imports):
        signature = set()
        try:
            if "peid_signatures" in pe_imports and pe_imports.get("peid_signatures") is not None:
                peid_signatures = pe_imports.get("peid_signatures")
                for each_signature in peid_signatures:
                    for sigentry in each_signature:
                        signature.add(sigentry.lower())
        except Exception as e:
            self.log.error(e)
        return signature

    def imports(self, pe_imports):
        dll_set = set()
        fn_names = set()
        try:
            if "imports" in pe_imports and pe_imports.get("imports") is not None:
                imports = pe_imports.get("imports")
                for each_import in imports:
                    import_names = each_import.get("imports")
                    dll_name = each_import.get("dll")
                    for each_import_name in import_names:
                        fn_names.add(each_import_name.get("name").lower())
                    dll_set.add(dll_name.lower())
        except Exception as e:
            self.log.error(e)
        return dll_set, fn_names

    def dirents(self, pe_imports):
        dirent_name = set()
        try:
            if "dirents" in pe_imports and pe_imports.get("dirents") is not None:
                dirents = pe_imports.get("dirents")
                for each_dirent_entry in dirents:
                    dirent_name.add(each_dirent_entry.get("name"))
        except Exception as e:
            self.log.error(e)
        return dirent_name

    def sections(self, pe_imports):
        characteristic_set = set()
        try:
            if "sections" in pe_imports and pe_imports.get("sections") is not None:
                sections = pe_imports.get("sections")
                for each_section in sections:
                    characteristics = each_section.get("characteristics")
                    for each_char in characteristics.split("|"):
                        characteristic_set.add(each_char.lower())
        except Exception as e:
            self.log.error(e)
        return characteristic_set

    def static_res(self, md5, results):
        """
        Takes the static results peid_signatures, imports, dirents, sections
        {"md5":{"static":{"peid_signatures":peid_signatures, "imports":imports, "dirents":dirents, "sections":sections}}}
        :param md5:
        :param results:
        :return:
        """
        signature = set()
        dll_set = set()
        fn_names = set()
        dirent_name = set()
        characteristic_set = set()
        try:
            if "static" in results and results.get("static") is not None:
                static_values = results.get("static")
                if "pe" in static_values and static_values.get("pe") is not None:
                    pe_imports = static_values.get("pe")

                    signature = self.peid_signatures(pe_imports)
                    dll_set, fn_names = self.imports(pe_imports)
                    dirent_name = self.dirents(pe_imports)
                    characteristic_set = self.sections(pe_imports)

        except Exception as e:
            self.log.error(e)

        inner_dict = dict()
        inner_dict["peid_signatures"] = signature
        inner_dict["dlls"] = dll_set
        inner_dict["fn_name"] = fn_names
        inner_dict["dirents"] = dirent_name
        inner_dict["sections"] = characteristic_set

        outer_dict = dict()
        outer_dict["static"] = inner_dict

        self.static_dump[md5] = outer_dict

    def proc_mem(self, md5, results):
        if len(results.get("procmemory")) > 0:
            self.alert_unknown_features(md5=md5, name=self.name)

    def decompression(self, md5, results):
        if len(results.get("decompression")) > 0:
            self.alert_unknown_features(md5=md5, name=self.name)

    def malheur(self, md5, results):
        """
        In this method, we take the malware family and score given by Malheur.
        {"md5":{"malheur":["malfamily","malscore"]}}
        :param results:
        :return:
        """
        malheur = list()
        try:
            if "malfamily" in results and results.get("malfamily") is not None:
                malheur.append(results.get("malfamily"))
            if "malscore" in results and results.get("malscore") is not None:
                malheur.append(results.get("malscore"))
        except Exception as e:
            self.log.error(e)

        inner_dict = dict()
        inner_dict["malheur"] = malheur

        self.malheur_dump[md5] = inner_dict

    def signatures(self, md5, results):
        """
        The order will be Name, Family, Description, References, confidence, weight, severity.
        {"md5":{"signatures": [name, family, description, references, confidence, weight, severity]}}
        :param md5:
        :param results:
        :return:
        """
        temp = defaultdict(list)
        try:
            if "signatures" in results and results.get("signatures") is not None:
                sign = results.get("signatures")
                name = []
                families = []
                description = []
                references = []
                confidence = []
                weight = []
                severity = []
                for sig in range(len(sign)):
                    if "confidence" in sign[sig] and sign[sig].get("confidence") is not None:
                        confidence = sign[sig].get("confidence")
                    if "description" in sign[sig] and sign[sig].get("description") is not None:
                        description = sign[sig].get("description")
                    if "name" in sign[sig] and sign[sig].get("name") is not None:
                        name = sign[sig].get("name")
                    if "severity" in sign[sig] and sign[sig].get("severity") is not None:
                        severity = sign[sig].get("severity")
                    if "weight" in sign[sig] and sign[sig].get("weight") is not None:
                        weight = sign[sig].get("weight")
                    if "families" in sign[sig] and sign[sig].get("families") is not None:
                        families = sign[sig].get("families")
                    if "references" in sign[sig] and sign[sig].get("references") is not None:
                        references = sign[sig].get("references")

                    temp[sig].append(name)
                    temp[sig].append(families)
                    temp[sig].append(description)
                    temp[sig].append(references)
                    temp[sig].append(confidence)
                    temp[sig].append(weight)
                    temp[sig].append(severity)
        except Exception as e:
            self.log.error(e)

        inner_dict = dict()
        inner_dict["signatures"] = temp
        self.signature_dump[md5] = inner_dict

    def statistics(self, md5, results):
        """
        This method has the stats of signatures, processing times and reporting times.
        We take the signatures from here and use them as Malware Evolutionary Labels.
        {"md5":{"statSignatures":thisMethodResults}}
        :param md5:
        :param results:
        :return:
        """
        signature_set = set()
        try:
            if "statistics" in results and results.get("statistics") is not None:
                stats = results.get("statistics")
                if "signatures" in stats and stats.get("signatures") is not None:
                    signatures = stats.get("signatures")
                    for each_sign in signatures:
                        signature_set.add(each_sign.get("name").lower())
        except Exception as e:
            self.log.error(e)

        inner_dict = dict()
        inner_dict["statSignatures"] = signature_set
        self.stats_dump[md5] = inner_dict

    def udp(self, network):
        """
        Takes the networkResults and returns the udp segment
        :param network:
        :return UDPDict:
        """
        src_set = set()
        dst_set = set()
        sport_set = set()
        dport_set = set()
        offset_set = set()
        try:
            if "udp" in network and network.get("udp") is not None:
                value = network.get("udp")

                for each_udp_value in value:
                    if "src" in each_udp_value and each_udp_value.get("src") is not None:
                        src_set.add(each_udp_value.get("src").lower())
                    if "dst" in each_udp_value and each_udp_value.get("dst") is not None:
                        dst_set.add(each_udp_value.get("dst").lower())
                    if "sport" in each_udp_value and each_udp_value.get("sport") is not None:
                        sport_set.add(each_udp_value.get("sport"))
                    if "dport" in each_udp_value and each_udp_value.get("dport") is not None:
                        dport_set.add(each_udp_value.get("dport"))
                    if "offset" in each_udp_value and each_udp_value.get("offset") is not None:
                        offset_set.add(each_udp_value.get("offset"))
        except Exception as e:
            self.log.error(e)

        udp = dict()
        udp["src"] = src_set
        udp["dst"] = dst_set
        udp["sport"] = sport_set
        udp["dport"] = dport_set
        udp["offset"] = offset_set
        return udp

    def hosts(self, network):
        """
        Takes the networkResults and returns the hosts segment
        :param network:
        :return:
        """
        country_name = set()
        hostname = set()
        ip = set()
        try:
            if "hosts" in network and network.get("hosts") is not None:
                hosts = network.get("hosts")
                if "country_name" in hosts and hosts.get("country_name") != "":
                    country_name.add(hosts.get("country_name").lower())
                if "hostname" in hosts and hosts.get("hostname") != "":
                    hostname.add(hosts.get("hostname").lower())
                if "ip" in hosts and hosts.get("ip") != "":
                    ip.add(hosts.get("ip").lower())
        except Exception as e:
            self.log.error(e)

        hosts = dict()
        hosts["country_name"] = country_name
        hosts["hostname"] = hostname
        hosts["ip"] = ip
        return hosts

    def dns(self, network):
        """
        Takes the network results and gives the dns segment.
        :param network:
        :return:
        """
        request = set()
        type = set()
        try:
            # TODO : Dns has answers field and I dont know what it is. Hence need to add to list of unknownfeatures
            if "dns" in network and network.get("dns") is not None:
                dns = network.get("dns")
                for each_dns_entry in dns:
                    if "type" in each_dns_entry and each_dns_entry.get("type") is not None:
                        type.add(each_dns_entry.get("type").lower())
                    if "request" in each_dns_entry and each_dns_entry.get("request") is not None:
                        request.add(each_dns_entry.get("request").lower())
        except Exception as e:
            self.log.error(e)

        dns = dict()
        dns["request"] = request
        dns["type"] = type
        return dns

    def domains(self, network):
        """
        Takes the network result and gives the domains segment.
        :param network:
        :return:
        """
        ip = set()
        dom = set()
        try:
            if "domains" in network and network.get("domains") is not None:
                domains = network.get("domains")
                for each_domain in domains:
                    if "ip" in each_domain and each_domain.get("ip") is not None:
                        ip.add(each_domain.get("ip").lower())
                    if "domain" in each_domain and each_domain.get("domain") is not None:
                        dom.add(each_domain.get("domain").lower())
        except Exception as e:
            self.log.error(e)

        domains = dict()
        domains["ip"] = ip
        domains["domain"] = dom

        return domains

    def network(self, md5, results):
        """
        The network module contains the following features
        1.'udp', 2.'http', 3.'smtp', 4.'tcp', 5.'sorted_pcap_sha256',
        6.'icmp', 7.'hosts', 8.'pcap_sha256', 9.'dns', 10.'domains', 11.'irc'
        Of the lot we are considering only udp, hosts, dns, domains.
        The rest can be considered if we are getting more improved results.
        {"md5": {"network": {"udp":udp, "hosts":hosts, "dns":dns, "domains":domains}}}
        :param md5:
        :param results:
        :return:
        """

        inner_dict = dict()
        if "network" in results and results.get("network") is not None:
            network = results.get("network")
            inner_dict["udp"] = self.udp(network)
            inner_dict["hosts"] = self.hosts(network)
            inner_dict["dns"] = self.dns(network)
            inner_dict["domains"] = self.domains(network)

        outer_dict = dict()
        outer_dict["network"] = inner_dict
        self.network_dump[md5] = outer_dict

    def file_fv(self, summary):
        """
        Takes the behavior summary and gives the fileset.
        :param summary:
        :return:
        """
        file_set = set()
        try:
            if "files" in summary and summary.get("files") is not None:
                files = summary.get("files")
                for each_file in files:
                    file_set.add(each_file.lower())

            if "write_files" in summary and summary.get("write_files") is not None:
                write_files = summary.get("write_files")
                for each_file in write_files:
                    file_set.add(each_file.lower())

            if "delete_files" in summary and summary.get("delete_files") is not None:
                delete_files = summary.get("delete_files")
                for each_file in delete_files:
                    file_set.add(each_file.lower())

            if "read_files" in summary and summary.get("read_files") is not None:
                read_files = summary.get("read_files")
                for each_file in read_files:
                    file_set.add(each_file.lower())
        except Exception as e:
            self.log.error(e)

        return file_set

    def keys_fv(self, summary):
        """
        Takes the behavior summary and returns the keyset
        :param summary:
        :return:
        """
        key_set = set()
        try:
            if "keys" in summary and summary.get("keys") is not None:
                keys = summary.get("keys")
                for each_key in keys:
                    key_set.add(each_key.lower())

            if "write_keys" in summary and summary.get("write_keys") is not None:
                write_keys = summary.get("write_keys")
                for each_key in write_keys:
                    key_set.add(each_key.lower())

            if "read_keys" in summary and summary.get("read_keys") is not None:
                read_keys = summary.get("read_keys")
                for each_key in read_keys:
                    key_set.add(each_key.lower())

            if "delete_keys" in summary and summary.get("delete_keys") is not None:
                delete_keys = summary.get("delete_keys")
                for each_key in delete_keys:
                    key_set.add(each_key.lower())
        except Exception as e:
            self.log.error(e)

        return key_set

    def services_fp(self, summary):
        """
        Takes the behavior summary and returns the services accessed part.
        :param summary:
        :return:
        """
        service_set = set()
        try:
            if "started_services" in summary and summary.get("started_services") is not None:
                started_services = summary.get("started_services")
                for each_service in started_services:
                    service_set.add(each_service.lower())

            if "created_services" in summary and summary.get("created_services") is not None:
                created_services = summary.get("created_services")
                for each_service in created_services:
                    service_set.add(each_service)
        except Exception as e:
            self.log.error(e)

        return service_set

    def mutexes(self, summary):
        mutex_set = set()
        try:
            if "mutexes" in summary and summary.get("mutexes") is not None:
                mutexes = summary.get("mutexes")
                for each_mutex in mutexes:
                    mutex_set.add(each_mutex.lower())
        except Exception as e:
            self.log.error(e)
        return mutex_set

    def executed_commands(self, summary):
        cmd_set = set()
        try:
            if "executed_commands" in summary and summary.get("executed_commands") is not None:
                executed_commands = summary.get("executed_commands")
                for each_cmd in executed_commands:
                    cmd_set.add(each_cmd.lower())
        except Exception as e:
            self.log.error(e)
        return cmd_set

    def resolved_apis(self, md5, summary):
        try:
            static_dict = self.static_dump.get(md5).get("static")
            dlls = static_dict.get("dlls")
            fn_name = static_dict.get("fn_name")

            if "resolved_apis" in summary and summary.get("resolved_apis") is not None:
                resolved_apis = summary.get("resolved_apis")
                for each_api in resolved_apis:
                    temp = each_api.split(".dll.")
                    dlls.add(temp[0])
                    fn_name.add(temp[1])

            static_dict["dlls"] = dlls
            static_dict["fn_name"] = fn_name

        except Exception as e:
            self.log.error(e)

    def behavior(self, md5, results):
        """
        There are a total of 13 sub-features here. We are considering all.
        'files', 'write_keys', 'keys', 'write_files', 'read_keys', 'delete_keys', 'delete_files',
        'mutexes', 'executed_commands', 'started_services', 'read_files', 'resolved_apis', 'created_services'
        Signature:
        {"md5":{"behavior":{
        "files": [files, write_files, delete_files, read_files],
        "keys": [keys, write_keys, read_keys, delete_keys],
        "mutexes": mutexes,
        "executed_commands": executed_commands,
        "resolved_apis": resolved_apis,
        "services": [started_services,created_service]}}}
        :param md5:
        :param results:
        :return:
        """
        inner_dict = dict()
        files_set = set()
        keys_set = set()
        service_set = set()
        mutex_set = set()
        cmd_set = set()
        try:
            if "behavior" in results and results.get("behavior") is not None:
                behavior = results.get("behavior")
                if "summary" in behavior and behavior.get("summary") is not None:
                    summary = behavior.get("summary")

                    files_set = self.file_fv(summary)
                    keys_set = self.keys_fv(summary)
                    service_set = self.services_fp(summary)
                    mutex_set = self.mutexes(summary)
                    cmd_set = self.executed_commands(summary)

                    self.resolved_apis(md5, summary)

        except Exception as e:
            self.log.error(e)
            summary = None

        inner_dict["files"] = files_set
        inner_dict["keys"] = keys_set
        inner_dict["summary"] = service_set
        inner_dict["mutexes"] = mutex_set
        inner_dict["executed_commands"] = cmd_set

        outer_dict = dict()
        outer_dict["behavior"] = inner_dict
        self.behavior_dump[md5] = outer_dict

    def get_cluster_path(self):
        """
        Gets the current path and stores the dumps in cluster
        :return:
        """
        _current_dir = os.path.abspath(os.path.dirname("__file__"))
        path = _current_dir + "/cluster/"
        self.log.info("Cluster Path: {0}".format(path))
        return path

    def write_dumps(self, md5):
        """
        This method will write the results into pickle dumps
        :return:
        """
        f1 = open(self.cluster_path + md5 + ".failed_analyses.cluster", "wa")
        pi.dump(self.failed_analyses, f1)
        f1.close()

        f2 = open(self.cluster_path + md5 + ".statsDump.cluster", "wa")
        pi.dump(self.stats_dump, f2)
        f2.close()

        f3 = open(self.cluster_path + md5 + ".signature_dump.cluster", "wa")
        pi.dump(self.signature_dump, f3)
        f3.close()

        f4 = open(self.cluster_path + md5 + ".malheur_dump.cluster", "wa")
        pi.dump(self.malheur_dump, f4)
        f4.close()

        f5 = open(self.cluster_path + md5 + ".static_dump.cluster", "wa")
        pi.dump(self.static_dump, f5)
        f5.close()

        f6 = open(self.cluster_path + md5 + ".network_dump.cluster", "wa")
        pi.dump(self.network_dump, f6)
        f6.close()

        f7 = open(self.cluster_path + md5 + ".unknownFeatures.cluster", "wa")
        pi.dump(self.unknown_features, f7)
        f7.close()

        f8 = open(self.cluster_path + md5 + ".behavior_dump.cluster", "wa")
        pi.dump(self.behavior_dump, f8)
        f8.close()

    def generate_meta_fp(self, results):
        """
        Generate the Meta fingerprint for all variants of the malware analyzed in current session.
        :return:
        """
        try:
            self.cluster_path = self.get_cluster_path()
            if not os.path.exists(self.cluster_path):
                os.makedirs(self.cluster_path)
            self.get_name(results)
            self.prepare_dumps()
            if not self.debug(self.name, results):
                self.static_res(self.name, results)
                self.proc_mem(self.name, results)
                self.decompression(self.name, results)
                self.malheur(self.name, results)
                self.signatures(self.name, results)
                self.statistics(self.name, results)
                self.network(self.name, results)
                self.behavior(self.name, results)

            self.write_dumps(self.name)

        except Exception as e:
            self.log.error(e)

    def prepare_dumps(self):
        """
        After every iteration we need to set the dumps to empty.
        :return:
        """
        # TODO : This is a fix, need to optimize this.
        self.stats_dump = defaultdict(dict)  # Is a dict of dict
        self.signature_dump = defaultdict(dict)  # Is a dict of dict
        self.malheur_dump = defaultdict(dict)  # Is a dict of dict
        self.static_dump = defaultdict(dict)  # Is a dict of dict
        self.network_dump = defaultdict(dict)  # Is a dict of dict
        self.behavior_dump = defaultdict(dict)  # Is a dict of dict
        self.unknown_features = defaultdict(list)  # Is a dict of list
