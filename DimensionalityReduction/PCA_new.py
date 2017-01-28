import pickle as pi

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import IncrementalPCA

matplotlib.use('Agg')


def prepare_data():
    f = open("mycsvfile.csv")
    l = list(list())
    names = list()

    for lines in f.readlines():
        split = lines.split(",")
        names.append(split[0])
        l.append(list(split[1][:-2]))

    input_matrix = np.array(l)
    return input_matrix


def singular_value_decomposition(input_matrix):
    (U, SIGMA, VT) = svd(input_matrix, full_matrices=False)
    return U, SIGMA, VT


def incremental_pca(input_matrix, n_components):
    """
    Assumption : batch_size should be equal to the number of components.
    """
    ipca = IncrementalPCA(n_components=n_components, batch_size=n_components)
    ipca.fit(input_matrix)
    reduced_matrix = ipca.transform(input_matrix)
    print("Shape of the reduced matrix is : {}".format(reduced_matrix.shape))
    return reduced_matrix


def compute_threshold(input_matrix, value):
    threshold = sum(input_matrix) * value
    index = len(input_matrix)
    for x in xrange(len(input_matrix)):
        if sum(input_matrix[:x]) > threshold:
            index = x
            break
    return index


def plotting(reduced_matrix):
    plt.plot(np.array(reduced_matrix[:, 0]), np.array(reduced_matrix[:, 1]), "ro")
    plt.savefig("plot.png")


if __name__ == "__main__":
    try:
        input_matrix = pi.load(open("input_matrix"))
    except Exception as e:
        print("The error is : {} ".format(e))
        input_matrix = prepare_data()

    U, Sigma, VT = singular_value_decomposition(input_matrix)
    index = compute_threshold(Sigma, 0.5)
    reduced_matrix = incremental_pca(input_matrix, index)
    plotting(reduced_matrix)
