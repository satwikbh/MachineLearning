import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from time import time

from Utils.LoggerUtil import LoggerUtil


class Plotting:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    @staticmethod
    def plot_it_2d(X, Y):
        t1 = time()
        
        fig = plt.figure(figsize=(60, 10))
        ax = fig.add_subplot(2, 5, 10)
        plt.title("Plotting (%.2g sec)" % (time() - t1))
        plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.get_cmap("jet", 10))
        plt.colorbar(ticks=range(10))
        plt.clim(-0.5, 9.5)
        plt.axis('tight')

        return plt

    @staticmethod
    def plot_it_3d(X, Y):
        t1 = time()

        fig = plt.figure(figsize=(50, 25))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.get_cmap("jet", 10))
        plt.title("Plotting (%.2g sec)" % (time() - t1))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.mouse_init()

        return plt
