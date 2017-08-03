import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import NullFormatter
from time import time

from Utils.LoggerUtil import LoggerUtil


class Plotting:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    @staticmethod
    def plot_it_2d(Y):
        t1 = time()
        fig = plt.figure(figsize=(60, 10))
        ax = fig.add_subplot(2, 5, 10)
        plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
        plt.title("t-SNE (%.2g sec)" % (time() - t1))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
        return plt

    @staticmethod
    def plot_it_3d(Y):
        t1 = time()

        fig = plt.figure(figsize=(50, 25))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], marker='o')
        plt.title("t-SNE (%.2g sec)" % (time() - t1))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.mouse_init()

        return plt
