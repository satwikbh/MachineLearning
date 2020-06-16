import glob
import json
from collections import defaultdict
from time import time

import marisa_trie

from HelperFunctions.HelperFunction import HelperFunction
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class TrieBasedPruning:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.helper = HelperFunction()
        self.files_pool = None
        self.reg_keys_pool = None
        self.mutex_pool = None
        self.exec_commands_pool = None
        self.network_pool = None
        self.static_feature_pool = None
        self.stat_sign_pool = None

    @staticmethod
    def build_trie(d):
        trie = marisa_trie.Trie(d.keys())
        return trie

    @staticmethod
    def prune_prefix_ids(key, l):
        if "" in l:
            l.remove("")
        if len(l) == 1:
            return l
        if key in l:
            l.remove(key)
        return l

    def get_prefix_ids(self, trie, d):
        meta_dict = defaultdict()
        meta_dict.default_factory = meta_dict.__len__
        for index, key in enumerate(d.keys()):
            if index % 10 ** 5 == 0:
                self.log.info(F"Iter : #{index}")
            unpruned_list = trie.prefixes(key)
            unpruned_list = self.prune_prefix_ids(key, unpruned_list)
            for _ in unpruned_list:
                meta_dict[_]
        return meta_dict

    def load_indi_feature_pools(self, individual_feature_pool_path):
        try:
            self.log.info("Loading files feature cluster")
            self.files_pool = json.load(open(individual_feature_pool_path + "/" + "files.json"))

            self.log.info("Loading registry keys feature cluster")
            self.reg_keys_pool = json.load(open(individual_feature_pool_path + "/" + "reg_keys.json"))

            self.log.info("Loading mutexes feature cluster")
            self.mutex_pool = json.load(open(individual_feature_pool_path + "/" + "mutexes.json"))

            self.log.info("Loading executed commands feature cluster")
            self.exec_commands_pool = json.load(open(individual_feature_pool_path + "/" + "executed_commands.json"))

            self.log.info("Loading network feature cluster")
            self.network_pool = json.load(open(individual_feature_pool_path + "/" + "network.json"))

            self.log.info("Loading static feature cluster")
            self.static_feature_pool = json.load(open(individual_feature_pool_path + "/" + "static_features.json"))

            self.log.info("Loading stat signature feature cluster")
            self.stat_sign_pool = json.load(open(individual_feature_pool_path + "/" + "stat_sign_features.json"))
        except Exception as e:
            self.log.error(F"Error : {e}")

    @staticmethod
    def load_tries(trie_path):
        files_trie_dict = marisa_trie.Trie()
        files_trie_dict.load(trie_path + "/" + "FilesTrie.dump")

        reg_keys_trie_dict = marisa_trie.Trie()
        reg_keys_trie_dict.load(trie_path + "/" + "RegKeysTrie.dump")

        executed_cmds_trie_dict = marisa_trie.Trie()
        executed_cmds_trie_dict.load(trie_path + "/" + "ExecCmdsTrie.dump")

        mutex_trie_dict = marisa_trie.Trie()
        mutex_trie_dict.load(trie_path + "/" + "MutexTrie.dump")

        return files_trie_dict, reg_keys_trie_dict, executed_cmds_trie_dict, mutex_trie_dict

    def prune_feature_pools(self, files_trie_dict, reg_keys_trie_dict, executed_cmds_trie_dict, mutex_trie_dict):
        meta_files_pool = self.get_prefix_ids(trie=files_trie_dict, d=self.files_pool)
        meta_reg_keys_pool = self.get_prefix_ids(trie=reg_keys_trie_dict, d=self.reg_keys_pool)
        meta_executed_cmds_pool = self.get_prefix_ids(trie=executed_cmds_trie_dict, d=self.exec_commands_pool)
        meta_mutex_pool = self.get_prefix_ids(trie=mutex_trie_dict, d=self.mutex_pool)
        return meta_files_pool, meta_reg_keys_pool, meta_executed_cmds_pool, meta_mutex_pool

    def save_feature_pools(self, **kwargs):
        pruned_indi_feature_pool_path = kwargs['pruned_indi_feature_pool_path']
        meta_files_pool = kwargs['meta_files_pool']
        meta_reg_keys_pool = kwargs['meta_reg_keys_pool']
        meta_executed_cmds_pool = kwargs['meta_executed_cmds_pool']
        meta_mutex_pool = kwargs['meta_mutex_pool']

        files_path = pruned_indi_feature_pool_path + "/" + "files.json"
        json.dump(meta_files_pool, open(files_path, "w"))

        reg_keys_path = pruned_indi_feature_pool_path + "/" + "reg_keys.json"
        json.dump(meta_reg_keys_pool, open(reg_keys_path, "w"))

        executed_commands_path = pruned_indi_feature_pool_path + "/" + "executed_commands.json"
        json.dump(meta_executed_cmds_pool, open(executed_commands_path, "w"))

        mutexes_path = pruned_indi_feature_pool_path + "/" + "mutexes.json"
        json.dump(meta_mutex_pool, open(mutexes_path, "w"))

        network_path = pruned_indi_feature_pool_path + "/" + "network.json"
        json.dump(self.helper.dict_values_to_list(self.network_pool.keys()), open(network_path, "w"))

        static_features_path = pruned_indi_feature_pool_path + "/" + "static_features.json"
        json.dump(self.helper.dict_values_to_list(self.static_feature_pool.keys()), open(static_features_path, "w"))

        stat_sign_features_path = pruned_indi_feature_pool_path + "/" + "stat_sign_features.json"
        json.dump(self.helper.dict_values_to_list(self.stat_sign_pool.keys()), open(stat_sign_features_path, "w"))

    def main(self):
        """
        Loads the individual feature pools and
        :return:
        """
        start_time = time()
        individual_feature_pool_path = self.config["data"]["individual_feature_pool_path"]
        pruned_indi_feature_pool_path = self.config["data"]["pruned_indi_feature_pool_path"]
        trie_path = self.config["data"]["trie_path"]

        self.load_indi_feature_pools(individual_feature_pool_path)
        if len(glob.glob(trie_path + "/" + "*.dump")) == 4:
            files_trie_dict, reg_keys_trie_dict, executed_cmds_trie_dict, mutex_trie_dict = self.load_tries(trie_path)
        else:
            self.log.info(F"Trie's not found at path : {trie_path}\nBuidling again")
            files_trie_dict = self.build_trie(self.files_pool)
            reg_keys_trie_dict = self.build_trie(self.reg_keys_pool)
            executed_cmds_trie_dict = self.build_trie(self.exec_commands_pool)
            mutex_trie_dict = self.build_trie(self.mutex_pool)

        meta_files_pool, meta_reg_keys_pool, meta_executed_cmds_pool, meta_mutex_pool = self.prune_feature_pools(
            files_trie_dict, reg_keys_trie_dict, executed_cmds_trie_dict, mutex_trie_dict)
        self.save_feature_pools(pruned_indi_feature_pool_path=pruned_indi_feature_pool_path,
                                meta_files_pool=meta_files_pool,
                                meta_reg_keys_pool=meta_reg_keys_pool,
                                meta_executed_cmds_pool=meta_executed_cmds_pool,
                                meta_mutex_pool=meta_mutex_pool)
        self.log.info(F"Total time taken : {time() - start_time}")


if __name__ == '__main__':
    trie_based_pruning = TrieBasedPruning()
    trie_based_pruning.main()
