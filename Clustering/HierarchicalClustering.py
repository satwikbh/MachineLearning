import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs


def optimalK(data, nrefs=3, max_clusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarray of shape (n_samples, n_features)
        nrefs: number of random sample data sets to produce
        maxClusters: maximum number of clusters to look for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, max_clusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, max_clusters)):

        # Holder for reference dispersion results
        ref_disps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            random_reference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(random_reference)

            ref_disp = km.inertia_
            ref_disps[i] = ref_disp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        orig_disp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(ref_disps)) - np.log(orig_disp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
    return gaps.argmax() + 1, resultsdf


def main():
    x, y = make_blobs(750, n_features=2, centers=12)

    """
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()
    """

    k, gapdf = optimalK(x, nrefs=5, max_clusters=15)
    print('Optimal k is: {} '.format(k))

    plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
    plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
    plt.grid(True)
    plt.xlabel('Cluster Count')
    plt.ylabel('Gap Value')
    plt.title('Gap Values by Cluster Count')
    plt.show()


if __name__ == '__main__':
    main()
