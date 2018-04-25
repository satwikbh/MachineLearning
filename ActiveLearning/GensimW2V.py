import os
import pickle as pi
import gensim

from time import time
from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil


class LoadVecs:
    """
    Custom class to load the feature pool element-wise.
    """

    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for pool_part_fname in open(os.path.join(self.dirname, fname)):
                pool_part = pi.load(pool_part_fname)
                for sentence in pool_part:
                    yield sentence


class GensimW2V:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()

    @staticmethod
    def load_params(load_vecs):
        """
        Parameters can be tuned from this method.
        :param load_vecs:
        :return:
        """
        params = dict()
        params['size'] = 200
        params['window'] = 50
        params['min_count'] = 10
        params['workers'] = 20

        params['load_vecs'] = load_vecs
        return params

    def perform_w2v(self, kwargs):
        """
        Perform w2v on the data.
        :param kwargs:
        :return:
        """
        self.log.info("Performing w2v on the data")
        model = gensim.models.Word2Vec(kwargs['load_vecs'],
                                       size=kwargs['size'], window=kwargs['window'],
                                       min_count=kwargs['min_count'], workers=kwargs['workers'])
        model.train(kwargs['load_vecs'], epochs=20)
        return model

    def save_model(self, w2v_results_path, model):
        """
        Save the w2v model at the given path.
        :param w2v_results_path:
        :param model:
        :return:
        """
        w2v_fname = w2v_results_path + "/" + "malware_w2v_embeds"
        self.log.info("Saving the w2v model at : {}".format(w2v_fname))
        model.wv.save_word2vec_format(fname=w2v_fname, binary=True)

    def main(self):
        start_time = time()
        self.log.info("Performing W2V")

        feature_pool_path = self.config["data"]["feature_pool_path"]
        w2v_results_path = self.config["models"]["w2v"]["results_path"]

        load_vecs = LoadVecs(dirname=feature_pool_path)
        params = self.load_params(load_vecs)
        model = self.perform_w2v(params)
        self.save_model(w2v_results_path, model)

        w2v_fname = w2v_results_path + "/" + "malware_w2v_embeds"
        model.wv.save_word2vec_format(fname=w2v_fname, binary=True)
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    w2v = GensimW2V()
    w2v.main()
