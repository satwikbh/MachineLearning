import hickle as hkl

from sklearn.manifold import LocallyLinearEmbedding
from time import time

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from HelperFunctions.HelperFunction import HelperFunction
from HelperFunctions.Plotting import Plotting


class LLE:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.config = ConfigUtil().get_config_instance()
        self.plot = Plotting()

    def plot_matrix(self, Y, iteration, plot_path):
        plt = self.plot.plot_it_2d(Y)
        plt.savefig(plot_path + "/" + "lle_2d_" + str(iteration) + ".png")
        plt = self.plot.plot_it_3d(Y)
        plt.savefig(plot_path + "/" + "lle_3d_" + str(iteration) + ".png")

    def perform_lle(self, iteration, partial_matrix):
        self.log.info("LLE on iteration #{}".format(iteration))
        model = LocallyLinearEmbedding(n_components=3, n_neighbors=2, eigen_solver="auto", n_jobs=-1)
        Y = model.fit_transform(partial_matrix)
        return Y

    def main(self):
        start_time = time()
        pruned_path = self.config['data']['pruned_feature_vector_path']
        plot_path = self.config['plots']['tsne']
        # pruned_path = "/home/satwik/Documents/Research/MachineLearning/Data/pruned_fv_path/"
        # plot_path = "/home/satwik/Documents/Research/MachineLearning/Data/plots/lle/"
        list_of_files = self.helper.get_files_ends_with_extension(path=pruned_path, extension=".hkl")
        for index, each_file in enumerate(list_of_files):
            partial_matrix = hkl.load(open(each_file)).todense()
            Y = self.perform_lle(index, partial_matrix)
            self.plot_matrix(Y, index, plot_path)
            break
        self.log.info("Total time taken : {}".format(time() - start_time))


if __name__ == '__main__':
    lle = LLE()
    lle.main()
