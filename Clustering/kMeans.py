import pickle as pi

from sklearn.cluster import KMeans

reduced_matrix = pi.load(open("ReducedMatrix.dump"))
print(reduced_matrix.shape)

kmeans = KMeans(n_clusters=25, random_state=0).fit(reduced_matrix)
print(list(kmeans.labels_))
